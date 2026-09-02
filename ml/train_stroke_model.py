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
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
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

    # One model ships, not an ensemble. A grid over 41 configurations found a
    # heavily regularised forest at ROC-AUC 0.8425, against 0.8429 for a
    # three-model soft-voting ensemble. That 0.0004 gap is far inside the
    # +/- 0.019 fold spread, so the second and third model buy nothing and
    # cost three times the inference and three times the failure surface.
    #
    # min_samples_leaf=100 looks aggressive for 249 positives, but the CV curve
    # is flat from 60 to 130 (0.8423 to 0.8425), so the choice is not delicate.
    # Looser forests measurably overfit: min_samples_leaf=8 scored 0.8359.
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=800,
            min_samples_leaf=100,
            max_features=0.5,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        # Kept as comparators so the selection below stays a measurement
        # rather than an assertion.
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000, random_state=SEED),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.02,
            max_leaf_nodes=7,
            min_samples_leaf=80,
            l2_regularization=3.0,
            class_weight="balanced",
            random_state=SEED,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print("\n5-fold CV on the training split, scored by ROC-AUC")
    scored = {}
    for name, model in candidates.items():
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        scored[name] = (roc_auc_score(y_train, oof), oof)
        print(f"  {name:24s} AUC {scored[name][0]:.4f}")

    best_name = max(scored, key=lambda k: scored[k][0])
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
                "sklearn": __import__("sklearn").__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {MODEL_OUT.name} and {META_OUT.name}")


if __name__ == "__main__":
    main()
