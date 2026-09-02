"""Does this model actually predict, or does it only run?

Exit code 0 proves nothing. The model this repo shipped for months loaded
cleanly, answered every request with HTTP 200, and reported 95.1% accuracy
while catching 0 of 249 strokes. Every "is it running" check passed.

This script asks the other question. It refits a clone of the artifact under
repeated stratified cross-validation, so the numbers are out-of-fold rather
than scored on data the model already memorised, and it fails loudly on the
specific ways a model on a 4.9% positive class looks fine but is useless.

Usage:  python ml/verify_model.py [path/to/model.pkl]
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
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app import server  # noqa: E402

# A model must clear all of these to be called working.
MIN_AUC = 0.75          # 0.50 is a coin flip
MIN_RECALL = 0.50       # must catch most strokes, not zero
MIN_SPREAD = 0.20       # best case vs worst case must differ
MIN_UNIQUE = 50         # a handful of distinct outputs means it is not really scoring

LOW = dict(age=26, gender="Female", ever_married="No", hypertension="No", heart_disease="No",
           avg_glucose_level=75, bmi=21, work_type="Private", residence_type="Rural",
           smoking_status="never smoked")
HIGH = dict(age=82, gender="Male", ever_married="Yes", hypertension="Yes", heart_disease="Yes",
            avg_glucose_level=290, bmi=48, work_type="Self-employed", residence_type="Urban",
            smoking_status="smokes")


def build_matrix(df):
    rows = []
    for rec in df.to_dict("records"):
        out = server.preprocess_data({
            "age": rec["age"], "gender": rec["gender"], "ever_married": rec["ever_married"],
            "hypertension": "Yes" if rec["hypertension"] == 1 else "No",
            "heart_disease": "Yes" if rec["heart_disease"] == 1 else "No",
            "avg_glucose_level": rec["avg_glucose_level"], "bmi": rec["bmi"],
            "work_type": rec["work_type"], "residence_type": rec["Residence_type"],
            "smoking_status": rec["smoking_status"],
        })
        rows.append(out.iloc[0] if hasattr(out, "iloc") else out[0])
    return pd.DataFrame(rows).reset_index(drop=True).astype(float)


def score_profile(model, profile):
    X = server.preprocess_data(profile)
    return float(model.predict_proba(X.values if hasattr(X, "values") else X)[0][1])


def main(path=None):
    path = Path(path or ROOT / "stroke_prediction_model.pkl")
    model = joblib.load(path)
    meta = {}
    meta_path = ROOT / "model_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    threshold = float(meta.get("threshold", 0.5))

    df = pd.read_csv(ROOT / "healthcare-dataset-stroke-data.csv")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    X, y = build_matrix(df), df["stroke"].to_numpy()

    print(f"artifact   {path.name}  ({type(model).__name__})")
    print(f"data       {len(X)} rows, {X.shape[1]} features, {y.sum()} positives "
          f"({y.mean()*100:.2f}%)")
    print(f"threshold  {threshold:.4f}")
    print("\nrefitting a clone under repeated stratified 5-fold x3 CV...")

    # cross_val_predict needs a true partition, so repeats are run as three
    # separate 5-fold passes with different shuffles and then averaged.
    aucs, aps, recs, precs, oofs = [], [], [], [], []
    for seed in (42, 7, 2024):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        o = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        p_hat = (o >= threshold).astype(int)
        oofs.append(o)
        aucs.append(roc_auc_score(y, o))
        aps.append(average_precision_score(y, o))
        recs.append(recall_score(y, p_hat, zero_division=0))
        precs.append(precision_score(y, p_hat, zero_division=0))

    oof = np.mean(oofs, axis=0)
    auc, ap = float(np.mean(aucs)), float(np.mean(aps))
    rec, prec = float(np.mean(recs)), float(np.mean(precs))
    auc_sd = float(np.std(aucs))
    pred = (oof >= threshold).astype(int)
    low, high = score_profile(model, LOW), score_profile(model, HIGH)
    spread, uniq = high - low, len(np.unique(np.round(oof, 4)))
    baseline = 1 - y.mean()

    print(f"\n  out-of-fold ROC-AUC     {auc:.4f} +/- {auc_sd:.4f}   (coin flip 0.50)")
    print(f"  average precision       {ap:.4f}   (random {y.mean():.4f})")
    print(f"  recall at threshold     {rec:.4f}   {pred[y==1].sum()}/{y.sum()} strokes caught")
    print(f"  precision at threshold  {prec:.4f}")
    print(f"  accuracy at threshold   {(pred==y).mean():.4f}   (always-no baseline {baseline:.4f})")
    print(f"  probability range       {oof.min():.4f} to {oof.max():.4f}")
    print(f"  distinct probabilities  {uniq}")
    print(f"  low-risk profile        {low*100:.1f}%")
    print(f"  high-risk profile       {high*100:.1f}%   spread {spread*100:.1f} points")

    checks = [
        ("discriminates better than chance", auc > MIN_AUC, f"AUC {auc:.4f} > {MIN_AUC}"),
        ("catches most strokes", rec > MIN_RECALL, f"recall {rec:.4f} > {MIN_RECALL}"),
        ("beats random ranking", ap > y.mean() * 2, f"AP {ap:.4f} > {y.mean()*2:.4f}"),
        ("predicts the positive class at all", pred.sum() > 0, f"{pred.sum()} positive predictions"),
        ("separates best from worst case", spread > MIN_SPREAD, f"spread {spread:.4f} > {MIN_SPREAD}"),
        ("produces a real score, not buckets", uniq >= MIN_UNIQUE, f"{uniq} distinct values"),
    ]
    print("\n" + "-" * 64)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:36s} {detail}")
    print("-" * 64)

    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print(f"\nVERDICT: NOT USABLE. Failed: {', '.join(failed)}")
        return 1
    print("\nVERDICT: USABLE. The model discriminates on data it was not fitted to.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
