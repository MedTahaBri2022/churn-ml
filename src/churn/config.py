"""Paths, feature definitions and constants shared by training and serving."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("CHURN_DATA", ROOT / "data" / "raw" / "telco_churn.csv"))
MODEL_DIR = Path(os.environ.get("CHURN_MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.environ.get("CHURN_REPORT_DIR", ROOT / "reports"))

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

YES_NO = ["Yes", "No"]
CATEGORIES: dict[str, list] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL = list(CATEGORIES)
FEATURES = NUMERIC + CATEGORICAL

# Add-on services counted by the engineered feature `services_count`.
SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Business assumption used to pick the decision threshold: missing a customer
# who leaves costs five times more than a retention offer made for nothing.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
