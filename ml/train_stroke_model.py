"""Train the stroke risk model that the API actually serves.

Two rules this script exists to enforce:

1. Features come from `app.server.preprocess_data`, the same function the API
   calls at request time. Training and serving cannot drift apart, because
   there is only one implementation.
2. The dataset is 4.87% positive, so accuracy is a useless target: always
   answering "no stroke" scores 95.13%. Models are selected on ROC-AUC,
   trained with balanced class weights, and the decision threshold is fitted
   rather than left at 0.5.

Run:  python ml/train_stroke_model.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.base import clone
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import server  # noqa: E402  (import triggers model load; we only want preprocess_data)

SEED = 42
DATA = ROOT / "healthcare-dataset-stroke-data.csv"
MODEL_OUT = ROOT / "stroke_prediction_model.pkl"
META_OUT = ROOT / "model_metadata.json"


def build_features(df):
    """Feature matrix built through the serving code path, row by row."""
    rows = []
    for rec in df.to_dict("records"):
        payload = {
            "age": rec["age"],
            "gender": rec["gender"],
            "ever_married": rec["ever_married"],
            "hypertension": "Yes" if rec["hypertension"] == 1 else "No",
            "heart_disease": "Yes" if rec["heart_disease"] == 1 else "No",
            "avg_glucose_level": rec["avg_glucose_level"],
            "bmi": rec["bmi"],
            "work_type": rec["work_type"],
            "residence_type": rec["Residence_type"],
            "smoking_status": rec["smoking_status"],
        }
        out = server.preprocess_data(payload)
        rows.append(out.iloc[0] if hasattr(out, "iloc") else out[0])
    frame = pd.DataFrame(rows).reset_index(drop=True)
    return frame.astype(float)


def main():
    df = pd.read_csv(DATA)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    X = build_features(df)
    y = df["stroke"].to_numpy()
    print(f"rows {len(X)}, features {X.shape[1]}, positives {y.sum()} ({y.mean() * 100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Selection is on ROC-AUC *and* clinical responsiveness, because AUC alone
    # picks the wrong model here. Age is so dominant on this dataset that an
    # age-only model scores 0.8261 while all 21 features score 0.8197. A search
    # optimising AUC therefore rewards a heavily regularised tree that never
    # splits on the rare binary flags: the previous winner moved its prediction
    # by -0.02 points for hypertension and +0.04 for heart disease, despite
    # heart disease tripling the stroke rate within the same age band
    # (5.79% -> 17.65% for ages 50-70).
    #
    # Logistic regression is linear, so every feature contributes in proportion
    # to its coefficient and none can be silently dropped. It costs 0.0043 AUC
    # against the tree, well inside the +/- 0.019 fold spread, and responds
    # +1.69 and +1.00 points to those two flags instead of nothing.
    def calibrated(estimator):
        return CalibratedClassifierCV(estimator, method="sigmoid", cv=5)

    candidates = {
        "logistic_regression": calibrated(make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000, random_state=SEED),
        )),
        # Comparators, kept so selection stays a measurement.
        "random_forest": calibrated(RandomForestClassifier(
            n_estimators=600, min_samples_leaf=20, max_features=0.5,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        )),
        "hist_gradient_boosting": calibrated(HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.02, max_leaf_nodes=7, min_samples_leaf=80,
            l2_regularization=3.0, class_weight="balanced", random_state=SEED,
        )),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("\n5-fold CV on the training split, scored by ROC-AUC")
    scored = {}
    for name, model in candidates.items():
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        scored[name] = (roc_auc_score(y_train, oof), oof)
        print(f"  {name:24s} AUC {scored[name][0]:.4f}")

    # A model that ignores hypertension and heart disease is not usable here,
    # whatever its AUC. Rank only among candidates that respond to both.
    def responds_to_clinical_flags(model):
        probe = dict(age=55, gender="Male", ever_married="Yes", hypertension="No",
                     heart_disease="No", avg_glucose_level=100, bmi=25,
                     work_type="Private", residence_type="Urban",
                     smoking_status="never smoked")
        fitted = clone(model).fit(X_train, y_train)

        def p(payload):
            row = server.preprocess_data(payload)
            return fitted.predict_proba(row.values if hasattr(row, "values") else row)[0][1]

        baseline = p(probe)
        # Relative, not absolute: what matters is whether setting the flag
        # moves the estimate in proportion to the risk it carries, and the
        # baseline itself varies by model.
        htn = p({**probe, "hypertension": "Yes"}) / baseline - 1
        heart = p({**probe, "heart_disease": "Yes"}) / baseline - 1
        return htn, heart

    usable = {}
    for name in scored:
        htn, heart = responds_to_clinical_flags(candidates[name])
        # A floor, deliberately well below the real effect. Within ages 50-70
        # this dataset shows hypertension raising the stroke rate 5.95% ->
        # 10.88% (+83%) and heart disease 5.79% -> 17.65% (+205%). The chosen
        # model responds roughly +56% and +33%, so it still understates heart
        # disease: that flag is only 5.4% of rows and correlates with age, so
        # a linear fit hands much of its effect to age. Requiring the full
        # +205% would mean overfitting 276 cases.
        #
        # 20% is therefore not a target, it is the line below which a model is
        # ignoring the factor rather than underweighting it. It rejects the
        # gradient-boosted tree (+6.8% for heart disease) and the random forest
        # (+14.1%), both of which were effectively age-only models.
        ok = htn >= 0.20 and heart >= 0.20
        print(f"  {name:24s} hypertension {htn*100:+6.1f}%  heart disease {heart*100:+6.1f}%"
              f"  {'usable' if ok else 'REJECTED: barely responds to clinical risk factors'}")
        if ok:
            usable[name] = scored[name][0]

    if not usable:
        raise RuntimeError("No candidate responds to hypertension and heart disease.")

    best_name = max(usable, key=lambda k: usable[k])
    best_model = candidates[best_name]
    oof = scored[best_name][1]
    print(f"\nselected: {best_name}")

    # Threshold fitted on out-of-fold predictions, never on the test split.
    # Youden's J maximises (sensitivity + specificity - 1), which is the right
    # balance for a screening tool that must actually flag positives.
    fpr, tpr, thresholds = roc_curve(y_train, oof)
    threshold = float(thresholds[np.argmax(tpr - fpr)])
    print(f"threshold fitted out-of-fold: {threshold:.4f}")

    best_model.fit(X_train, y_train)
    proba = best_model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(y_test)),
        "specificity": float(tn / (tn + fp)),
    }

    print("\nheld-out test set (never seen during training or threshold fitting)")
    print(f"  ROC-AUC              {metrics['roc_auc']:.4f}")
    print(f"  average precision    {metrics['average_precision']:.4f}")
    print(f"  recall (strokes hit) {metrics['recall']:.4f}   {tp}/{tp + fn}")
    print(f"  precision            {metrics['precision']:.4f}")
    print(f"  specificity          {metrics['specificity']:.4f}")
    print(f"  accuracy             {metrics['accuracy']:.4f}   (baseline {1 - y_test.mean():.4f})")
    print(f"  confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  probability range    {proba.min():.4f} to {proba.max():.4f}")

    # A single 80/20 split of 50 positives is a noisy estimate. Report the
    # spread across 20 splits so the page cannot present one draw as precise.
    split_aucs, split_recalls = [], []
    for split_seed in range(20):
        a_tr, a_te, b_tr, b_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=split_seed
        )
        fitted = clone(best_model).fit(a_tr, b_tr)
        probs = fitted.predict_proba(a_te)[:, 1]
        split_aucs.append(roc_auc_score(b_te, probs))
        split_recalls.append(recall_score(b_te, (probs >= threshold).astype(int), zero_division=0))

    # Overall AUC is dominated by ranking across ages. Within an age band the
    # model is far weaker, which the headline number hides entirely.
    oof = cross_val_predict(
        clone(best_model), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    subgroups = {}
    raw = pd.read_csv(DATA)
    for label, mask in [
        ("gender=Male", (raw.gender == "Male").to_numpy()),
        ("gender=Female", (raw.gender == "Female").to_numpy()),
        ("age_40_60", ((raw.age >= 40) & (raw.age < 60)).to_numpy()),
        ("age_60_80", ((raw.age >= 60) & (raw.age < 80)).to_numpy()),
        ("age_80_plus", (raw.age >= 80).to_numpy()),
    ]:
        if y[mask].sum() >= 5:
            subgroups[label] = {
                "n": int(mask.sum()),
                "positives": int(y[mask].sum()),
                "roc_auc": float(roc_auc_score(y[mask], oof[mask])),
            }

    print("\nstability across 20 splits")
    print(f"  ROC-AUC  {np.mean(split_aucs):.4f} +/- {np.std(split_aucs):.4f}")
    print(f"  recall   {np.mean(split_recalls):.4f} +/- {np.std(split_recalls):.4f}")
    print("\nwithin-subgroup ROC-AUC (the headline number hides this)")
    for k, v in subgroups.items():
        print(f"  {k:16s} n={v['n']:5d} pos={v['positives']:3d}  AUC {v['roc_auc']:.4f}")

    # Ship a model fitted on everything, with the threshold and metrics measured above.
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_OUT)
    META_OUT.write_text(
        json.dumps(
            {
                "model": best_name,
                "threshold": threshold,
                "features": list(X.columns),
                "n_features": X.shape[1],
                "trained_on": DATA.name,
                "n_rows": int(len(X)),
                "positive_rate": float(y.mean()),
                "metrics_holdout": metrics,
                "majority_class_accuracy": float(1 - y.mean()),
                "stability_20_splits": {
                    "roc_auc_mean": float(np.mean(split_aucs)),
                    "roc_auc_sd": float(np.std(split_aucs)),
                    "recall_mean": float(np.mean(split_recalls)),
                    "recall_sd": float(np.std(split_recalls)),
                },
                "subgroup_roc_auc": subgroups,
                "calibrated": True,
                "brier_score": float(brier_score_loss(y_test, proba)),
                "sklearn": __import__("sklearn").__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {MODEL_OUT.name} and {META_OUT.name}")


if __name__ == "__main__":
    main()
