"""Guard against the failure this model already had once.

The previous artifact scored 95% accuracy by never predicting a stroke: AUC
0.56, recall 0.00, and every input landing between 0.13 and 0.21. Accuracy
alone hid it completely. These asserts fail if that ever comes back.

Run:  python ml/test_model_sanity.py
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

from app import server  # noqa: E402

LOW = dict(age=26, gender="Female", ever_married="No", hypertension="No", heart_disease="No",
           avg_glucose_level=75, bmi=21, work_type="Private", residence_type="Rural",
           smoking_status="never smoked")
HIGH = dict(age=82, gender="Male", ever_married="Yes", hypertension="Yes", heart_disease="Yes",
            avg_glucose_level=290, bmi=48, work_type="Self-employed", residence_type="Urban",
            smoking_status="smokes")


def score(profile):
    X = server.preprocess_data(profile)
    X = X.values if hasattr(X, "values") else X
    return float(server.models["main"].predict_proba(X)[0][1])


def main():
    assert server.models, "no models loaded"
    assert not server.using_mock_model, "serving the mock model, not a real one"

    meta = json.loads((ROOT / "model_metadata.json").read_text())
    m = meta["metrics_holdout"]

    assert m["roc_auc"] > 0.75, f"ROC-AUC {m['roc_auc']:.4f} too low; 0.50 is a coin flip"
    assert m["recall"] > 0.50, f"recall {m['recall']:.4f}: the model is missing most strokes"
    # The point is that the threshold was FITTED, not that it lands below 0.5.
    # An earlier version asserted < 0.5 and wrongly failed a perfectly good
    # gradient-boosted model whose fitted threshold was 0.5308; probability
    # distributions differ by model family. What matters is that nobody left it
    # at sklearn's untouched 0.5 default, and that recall above proves the
    # operating point actually flags cases.
    assert abs(meta["threshold"] - 0.5) > 1e-9, "threshold left at the untouched 0.5 default"

    low, high = score(LOW), score(HIGH)
    assert high > low, f"high-risk profile ({high:.3f}) must outscore low-risk ({low:.3f})"
    assert high - low > 0.20, f"spread {high - low:.3f} too flat to be discriminating"
    assert high >= meta["threshold"], "worst-case profile does not even get flagged"

    print(f"OK  model={meta['model']}  AUC={m['roc_auc']:.4f}  recall={m['recall']:.4f}")
    print(f"OK  low-risk {low * 100:.1f}%  high-risk {high * 100:.1f}%  spread {(high - low) * 100:.1f}pts")


if __name__ == "__main__":
    main()
    print("all model sanity checks passed")
