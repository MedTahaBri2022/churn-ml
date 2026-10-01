# Churn ML: from raw data to a monitored prediction API

An end-to-end machine learning project on customer churn: data validation,
feature engineering, model selection, experiment tracking, a model card, a
prediction API with input validation and metrics, and data drift detection.

**Python · scikit-learn · pandas · MLflow · FastAPI · Pydantic · Prometheus metrics · Docker · pytest**

[![CI](https://github.com/MedTahaBri2022/churn-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/MedTahaBri2022/churn-ml/actions/workflows/ci.yml)

## Results

Public Telco Customer Churn dataset: 7,043 customers, 26.5% churn.

| Model (5-fold CV on the training split) | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Baseline (class prior) | 0.500 | 0.265 |
| Logistic regression | 0.848 | 0.661 |
| Random forest | 0.848 | 0.664 |
| **Gradient boosting** (selected) | **0.849** | **0.668** |

Hold-out test set (1,409 customers, used once): **ROC-AUC 0.846**, PR-AUC 0.662,
Brier score 0.135.

| Decision threshold | Precision | Recall | Churners missed |
| --- | ---: | ---: | ---: |
| Default 0.50 | 0.681 | 0.519 | 180 of 374 |
| **Cost-based 0.19** | 0.460 | **0.853** | **55 of 374** |

Two honest readings of these numbers:

- **The three models are tied.** Their cross-validation scores differ by 0.001,
  less than the fold-to-fold variation. Gradient boosting is selected because
  the rule is "best mean ROC-AUC", but logistic regression would be an equally
  defensible, simpler choice.
- **The threshold matters more than the model.** Moving from 0.50 to a
  threshold chosen for the business cost catches 125 more churners in the test
  set, at the price of more retention offers sent for nothing.

The full report is generated at each training:
[reports/model_card.md](reports/model_card.md),
[reports/metrics.json](reports/metrics.json).

## Pipeline

```
data/raw/telco_churn.csv
        │  clean      blank TotalCharges of first-month customers -> 0
        │  validate   schema, categories, ranges, duplicate ids (all problems reported)
        ▼
 stratified split 80 / 20 ──────────────────────────────┐
        │                                               │ test set, untouched
        ▼                                               │
 scikit-learn Pipeline                                  │
   engineered features -> impute/scale + one-hot        │
   3 candidates x small grid, 5-fold CV                 │
        │                                               │
        ▼                                               │
 best model -> threshold from out-of-fold predictions   │
        │                                               ▼
        └────────────────────────────► final metrics, permutation importance
        │
        ▼
 models/  model.joblib · metadata.json · reference.json (for drift)
 reports/ metrics.json · model_card.md          MLflow run (params, metrics, artifacts)
        │
        ▼
 FastAPI  /predict · /predict/batch · /model · /health · /metrics
```

## Design decisions

- **No leakage, by construction.** Preprocessing lives inside the pipeline, so
  it is refitted on each training fold. Engineered features are row-wise only.
  The model, its hyper-parameters *and* the decision threshold are all chosen
  with cross-validation on the training split; the test set is used once.
- **One implementation of the features.** The API calls the same pipeline
  object that was trained; there is no second copy of the transformations to
  keep in sync.
- **Data is validated before it is trusted.** Unexpected categories, negative
  amounts, missing columns or duplicate ids stop the training with the complete
  list of problems.
- **A threshold with a reason.** It minimizes an explicit cost (a missed
  churner costs five times a wasted offer, configurable in `config.py`)
  instead of defaulting to 0.5.
- **Reproducible.** Fixed seed; a test trains twice and compares the metrics.
  The data file's SHA-256 and the git revision are stored with the model.
- **A quality gate.** `python -m churn check` fails CI if the test ROC-AUC
  drops below 0.82, and the Docker image cannot be built with a model under
  that bar.
- **The API refuses what the model was not trained on.** Pydantic types mirror
  the training schema (allowed categories, ranges, no extra fields): a bad
  request is a `422`, not a silent wrong prediction.
- **Drift is measurable.** A compact description of the training distribution
  is saved with the model; `python -m churn drift batch.csv` computes the
  Population Stability Index per feature and exits non-zero above 0.2, so it
  can run as a scheduled check.

## Run it

Requirements: Python 3.11+ (or only Docker).

```bash
pip install -e ".[dev]"

python -m churn train                 # CV leaderboard, artifacts, MLflow run
python -m churn check                 # quality gate
uvicorn churn.serve:app --port 8000   # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":2,
  "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
  "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
  "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":95.0,"TotalCharges":190.0}'
# {"churn_probability":0.7657,"churn_predicted":true,"threshold":0.19,"model":"gradient_boosting"}
```

With Docker (the image trains the model at build time, then serves it):

```bash
docker compose up --build
```

Experiment tracking is stored in `mlflow.db` (SQLite) with artifacts in
`mlartifacts/`. To browse runs, install the full `mlflow` package and run
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### Drift check

```
$ python -m churn drift new_customers.csv
Contract           psi=3.7362 DRIFT
MonthlyCharges     psi=2.8388 DRIFT
tenure             psi=0.5958 DRIFT
TotalCharges       psi=0.2511 DRIFT
TechSupport        psi=0.1682
...
4 feature(s) above 0.2: Contract, MonthlyCharges, TotalCharges, tenure
```

(Output for a simulated population of month-to-month customers paying 80%
more. A random sample of the original data reports no drift.)

## Tests

```bash
pytest
```

26 tests: cleaning and validation rules, stratified and reproducible split,
engineered features, unseen categories, model better than the baseline,
reproducible training, complete artifacts, cost-based threshold, drift
detected and not falsely detected, API predictions, rejected payloads, batch
limit, metrics, behaviour without a model.

CI runs the tests, trains the three candidates, applies the quality gate,
builds the Docker image and calls the running container.

## Repository layout

```
src/churn/
  config.py     features, categories, costs, paths
  data.py       load, clean, validate, split
  features.py   engineered features + preprocessing pipeline
  evaluate.py   metrics, cost-based threshold
  train.py      model selection, artifacts, model card, MLflow
  drift.py      Population Stability Index
  serve.py      FastAPI application
  __main__.py   command line: train / check / drift
tests/          pytest suite
reports/        generated metrics and model card
data/           dataset and its attribution
```

## Limits

- One static snapshot: no time-based validation, which is how a churn model
  is really judged.
- Probabilities are not recalibrated (Brier score 0.135); the threshold is
  chosen on the raw scores.
- `gender` and `SeniorCitizen` are used as features; no fairness audit.
- The drift check covers input features only, not the relationship between
  features and churn (concept drift needs labels).
- No model registry or automated retraining; MLflow is used for tracking only.

## Data

Telco Customer Churn sample dataset by IBM, Apache License 2.0. See
[data/README.md](data/README.md).

## License

MIT (code).
