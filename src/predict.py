"""
HealthRisk-AI | Step 4 — Predict new patients
==============================================
Loads the trained pipeline (models/heart_risk_model.joblib) and scores
example patients, printing a human-readable risk report.

Run:
    python src/predict.py
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_risk_model.joblib"

FEATURES = ["age", "sex", "systolic_bp", "cholesterol", "glucose", "bmi",
            "smoker", "diabetes", "family_history", "physical_activity",
            "max_heart_rate"]

# (label, age, sex, sbp, chol, glucose, bmi, smoker, diabetes, fam_hist,
#  activity h/wk, max heart rate)  — values follow FEATURES order above
EXAMPLE_PATIENTS = [
    ("Patient A — healthy profile",   42, 0, 118, 190,  92, 23.1, 0, 0, 0, 7.5, 178),
    ("Patient B — average profile",   55, 0, 132, 225, 100, 27.0, 0, 0, 1, 3.0, 160),
    ("Patient C — high-risk profile", 66, 1, 158, 288, 148, 33.4, 1, 1, 1, 0.5, 132),
]


def risk_band(prob: float) -> str:
    if prob < 0.30:
        return "LOW"
    if prob < 0.60:
        return "MODERATE"
    return "HIGH"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model, name = bundle["model"], bundle["model_name"]
    print(f"Loaded model : {name}")
    print("-" * 62)

    records = {}
    for label, *vals in EXAMPLE_PATIENTS:
        x = pd.DataFrame([vals], columns=FEATURES)
        p = float(model.predict_proba(x)[0, 1])
        band = risk_band(p)
        flag = {"LOW": "OK", "MODERATE": "WATCH", "HIGH": "ALERT"}[band]
        print(f"{label:<30} risk = {p:5.1%}  [{band:<8}] {flag}")
        records[label] = {"predicted_risk": f"{p:.1%}", "band": band, "flag": flag}

    print("-" * 62)
    print("\nAs DataFrame:")
    print(pd.DataFrame.from_dict(records, orient="index").to_string())


if __name__ == "__main__":
    main()
