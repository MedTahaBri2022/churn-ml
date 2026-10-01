import pytest
from fastapi.testclient import TestClient

from churn import config, serve


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", trained["model_dir"])
    with TestClient(serve.app) as test_client:
        yield test_client


def test_health_and_model_metadata(client):
    assert client.get("/health").json() == {"status": "ok", "model": "logistic_regression"}
    assert 0 < client.get("/model").json()["threshold"] < 1


def test_predicts_high_risk_for_a_typical_churner(client, customer):
    body = client.post("/predict", json=customer).json()

    assert body["churn_predicted"] is True
    assert body["churn_probability"] > 0.6


def test_predicts_low_risk_for_a_loyal_customer(client, customer):
    loyal = {
        **customer,
        "tenure": 70,
        "Contract": "Two year",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.0,
        "TotalCharges": 4200.0,
    }
    body = client.post("/predict", json=loyal).json()

    assert body["churn_predicted"] is False
    assert body["churn_probability"] < 0.2


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Lifetime"},
        {"tenure": -1},
        {"MonthlyCharges": "a lot"},
        {"favourite_colour": "blue"},
    ],
)
def test_invalid_payloads_are_refused(client, customer, change):
    assert client.post("/predict", json={**customer, **change}).status_code == 422


def test_missing_field_is_refused(client, customer):
    customer.pop("tenure")
    assert client.post("/predict", json=customer).status_code == 422


def test_batch_prediction_and_limit(client, customer):
    response = client.post("/predict/batch", json={"customers": [customer] * 3})
    assert response.status_code == 200
    assert len(response.json()) == 3

    too_many = client.post("/predict/batch", json={"customers": [customer] * 1001})
    assert too_many.status_code == 422


def test_metrics_count_predictions(client, customer):
    client.post("/predict", json=customer)
    text = client.get("/metrics").text

    assert 'churn_predictions_total{outcome="churn"}' in text
    assert "churn_prediction_seconds_count" in text


def test_without_a_model_the_api_says_so(tmp_path, monkeypatch, customer):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=customer).status_code == 503
