import pandas as pd
import pytest

from churn import config, data
from churn.train import train


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return data.load_dataset()


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """One quick training (single candidate) shared by the whole test session."""
    model_dir = tmp_path_factory.mktemp("models")
    report_dir = tmp_path_factory.mktemp("reports")
    report = train(
        model_dir=model_dir,
        report_dir=report_dir,
        only=["logistic_regression"],
        track=False,
    )
    return {"report": report, "model_dir": model_dir, "report_dir": report_dir}


@pytest.fixture()
def customer() -> dict:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
    }


def frame(customer: dict) -> pd.DataFrame:
    return pd.DataFrame([customer])[config.FEATURES]
