"""Prediction API.

    uvicorn churn.serve:app --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from . import config
from .train import load_artifacts, predict_frame

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]

PREDICTIONS = Counter("churn_predictions_total", "Predictions served", ["outcome"])
LATENCY = Histogram("churn_prediction_seconds", "Time spent computing predictions")
MAX_BATCH = 1000


class Customer(BaseModel):
    """One customer. Types and allowed values mirror the training schema, so a
    malformed request is refused (422) before it reaches the model."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100000)


class Prediction(BaseModel):
    churn_probability: float
    churn_predicted: bool
    threshold: float
    model: str


class Batch(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loaded once at start-up; a missing model is reported by /health instead
    # of crashing the process, so the cause is visible from outside.
    try:
        state["model"], state["metadata"], state["reference"] = load_artifacts(config.MODEL_DIR)
    except FileNotFoundError:
        state.clear()
    yield
    state.clear()


app = FastAPI(title="Churn prediction API", version="1.0.0", lifespan=lifespan)


def _predict(customers: list[Customer]) -> list[Prediction]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="No trained model is available")
    started = time.perf_counter()
    frame = pd.DataFrame([customer.model_dump() for customer in customers])
    probabilities = predict_frame(state["model"], frame)
    LATENCY.observe(time.perf_counter() - started)

    threshold = state["metadata"]["threshold"]
    results = []
    for probability in probabilities:
        churn = bool(probability >= threshold)
        PREDICTIONS.labels("churn" if churn else "stay").inc()
        results.append(
            Prediction(
                churn_probability=round(float(probability), 4),
                churn_predicted=churn,
                threshold=threshold,
                model=state["metadata"]["model_name"],
            )
        )
    return results


@app.get("/health")
def health(response: Response):
    if "model" not in state:
        response.status_code = 503
        return {"status": "no model", "hint": "run `python -m churn train` first"}
    return {"status": "ok", "model": state["metadata"]["model_name"]}


@app.get("/model")
def model_info():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="No trained model is available")
    return state["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: Batch):
    return _predict(batch.customers)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
