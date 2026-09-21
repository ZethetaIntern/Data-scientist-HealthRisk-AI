"""
HealthRisk-AI | Step 1 — Generate the dataset
==============================================
Creates a realistic, fully-reproducible synthetic heart-health dataset
(4,000 patient records). Feature ranges and risk relationships follow
well-known clinical patterns (Framingham-style risk factors):

    risk  ~  f(age, sex, systolic BP, cholesterol, glucose, BMI,
                smoking, diabetes, family history, physical activity)

Run:
    python src/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 4_000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_risk_data.csv"


def _clip(arr, low, high):
    return np.clip(arr, low, high)


def generate(n: int = N_SAMPLES, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Demographics & lifestyle
    # ------------------------------------------------------------------
    age = _clip(rng.normal(53, 13, n), 29, 79).round().astype(int)
    sex = rng.binomial(1, 0.52, n)                      # 1 = male
    smoker = rng.binomial(1, 0.24 + 0.08 * sex, n)      # males smoke more
    activity = _clip(np.abs(rng.normal(3.5, 2.6, n)), 0, 14).round(1)

    # ------------------------------------------------------------------
    # 2. Clinical measurements (correlated with age / lifestyle)
    # ------------------------------------------------------------------
    systolic_bp = _clip(
        rng.normal(124, 15, n) + 0.38 * (age - 50) + 4.0 * smoker, 92, 200
    ).round().astype(int)

    cholesterol = _clip(
        rng.normal(222, 42, n) + 0.55 * (age - 50), 130, 400
    ).round().astype(int)

    bmi = _clip(rng.normal(27.4, 4.6, n) + 0.05 * (age - 50), 16.5, 48).round(1)

    diabetes = rng.binomial(
        1, _clip(0.03 + 0.004 * (age - 40) + 0.012 * (bmi - 27), 0.01, 0.45)
    )

    glucose = _clip(
        rng.normal(96, 15, n) + 45 * diabetes, 68, 300
    ).round().astype(int)

    family_history = rng.binomial(1, 0.30, n)
    max_heart_rate = _clip(
        rng.normal(202 - 0.65 * age, 11, n) - 3 * smoker, 72, 205
    ).round().astype(int)
    hr_reserve = 202 - 0.65 * age - max_heart_rate   # lower achieved HR = fitter

    # ------------------------------------------------------------------
    # 3. Latent risk score -> probability -> outcome
    #    (includes a realistic age x smoking interaction)
    # ------------------------------------------------------------------
    logit = (
        -1.55
        + 0.055 * (age - 50)
        + 0.25 * sex
        + 0.028 * (systolic_bp - 125)
        + 0.020 * (cholesterol - 200)
        + 0.008 * (glucose - 95)
        + 0.085 * (bmi - 27)
        + 0.95 * smoker
        + 1.10 * diabetes
        + 0.55 * family_history
        - 0.18 * activity
        + 0.10 * hr_reserve                    # poor fitness raises risk
        + 0.045 * (age - 50) * smoker          # interaction term
    )
    p_risk = 1.0 / (1.0 + np.exp(-logit))
    heart_risk = rng.binomial(1, p_risk)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "glucose": glucose,
            "bmi": bmi,
            "smoker": smoker,
            "diabetes": diabetes,
            "family_history": family_history,
            "physical_activity": activity,
            "max_heart_rate": max_heart_rate,
            "heart_risk": heart_risk,
        }
    )
    return df


def main() -> None:
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    prevalence = df["heart_risk"].mean()
    print(f"Saved {len(df):,} records -> {OUT_PATH}")
    print(f"Risk prevalence          : {prevalence:.1%}")
    print(f"Mean age                 : {df['age'].mean():.1f}")
    print(f"Smokers                  : {df['smoker'].mean():.1%}")
    print(f"Diabetic                 : {df['diabetes'].mean():.1%}")
    print("\nPreview:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
