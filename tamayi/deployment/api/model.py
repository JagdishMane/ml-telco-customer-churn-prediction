"""Model loading, feature engineering, and scoring for the churn service.

This module is the reusable core of the API. It loads the trained artifacts once,
rebuilds the engineered features exactly as notebook 03 did, applies the saved
preprocessor, and runs the model. Keeping the field metadata here (RAW_FIELDS) gives
the schema, the /schema endpoint, and the optional LLM a single source of truth.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# Resolve the model directory. This file lives at tamayi/deployment/api/model.py, so
# the models folder is two levels up (tamayi/models). Allow an override via the
# MODEL_DIR environment variable for other deployment layouts.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(_DEFAULT_MODEL_DIR)))

# The seven raw fields the user supplies. CustomerID and Churn are intentionally
# excluded: the first is an identifier, the second is the target. Each entry carries
# enough metadata to build a form, validate input, and prompt the LLM.
RAW_FIELDS = [
    {"name": "Age", "type": "int", "min": 18, "max": 100,
     "description": "Customer age in years"},
    {"name": "Gender", "type": "category",
     "choices": ["Female", "Male", "Other"],
     "description": "Customer gender"},
    {"name": "Tenure", "type": "int", "min": 1, "max": 72,
     "description": "Months the customer has stayed with the company"},
    {"name": "MonthlyCharges", "type": "float", "min": 10.0, "max": 150.0,
     "description": "Monthly bill in USD"},
    {"name": "TotalCharges", "type": "float", "min": 0.0, "max": 11000.0,
     "description": "Total amount billed over the customer's tenure in USD"},
    {"name": "Contract", "type": "category",
     "choices": ["Month-to-month", "One year", "Two year"],
     "description": "Contract type"},
    {"name": "PaymentMethod", "type": "category",
     "choices": ["Bank transfer", "Credit card", "Electronic check", "Mailed check"],
     "description": "Preferred payment method"},
]

RAW_FIELD_NAMES = [f["name"] for f in RAW_FIELDS]

# Default decision threshold. Notebook 05 recommends a recall-favoring operating point
# around 0.40, because missing a churner costs more than a false alarm.
DEFAULT_THRESHOLD = 0.40


@lru_cache(maxsize=1)
def _load_artifacts():
    """Load the model and preprocessor once and cache them."""
    model_path = MODEL_DIR / "best_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Could not find model artifacts in {MODEL_DIR}. Expected best_model.pkl "
            "and preprocessor.pkl. Set the MODEL_DIR environment variable if they live "
            "elsewhere."
        )
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four engineered features, matching notebook 03 exactly."""
    out = df.copy()
    out["AvgChargePerMonth"] = out["TotalCharges"] / out["Tenure"]
    out["IsShortTenure"] = (out["Tenure"] < 12).astype(int)
    out["IsHighMonthly"] = (out["MonthlyCharges"] > 100).astype(int)
    out["SeniorFlag"] = (out["Age"] >= 60).astype(int)
    return out


def _risk_tier(probability: float) -> str:
    """Bucket a churn probability into a simple business risk tier."""
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Medium"
    return "Low"


def predict(records: list[dict], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Score one or more raw customer records.

    Each record is a dict with the seven raw fields. Returns one result dict per
    record with the churn probability, the churn flag at the given threshold, and a
    risk tier.
    """
    model, preprocessor = _load_artifacts()

    raw = pd.DataFrame(records)[RAW_FIELD_NAMES]
    engineered = engineer(raw)
    transformed = preprocessor.transform(engineered)

    # Wrap in a DataFrame with the post-transform column names so the model sees the
    # same feature names it was fitted with (avoids a sklearn feature-name warning).
    transformed = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
    probabilities = model.predict_proba(transformed)[:, 1]

    results = []
    for prob in probabilities:
        prob = float(prob)
        results.append({
            "churn": bool(prob >= threshold),
            "probability": round(prob, 4),
            "risk_tier": _risk_tier(prob),
            "threshold": threshold,
        })
    return results


def llm_configured() -> bool:
    """Whether the optional LLM is configured (an API key is present)."""
    return bool(os.getenv("LLM_API_KEY"))
