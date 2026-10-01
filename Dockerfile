# Two stages: the first trains the model, the second only serves it.
FROM python:3.12-slim AS train

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
# Deterministic (fixed seed), so the image always contains the same model.
ENV CHURN_DATA=/app/data/raw/telco_churn.csv \
    CHURN_MODEL_DIR=/app/models \
    CHURN_REPORT_DIR=/app/reports
RUN python -m churn train --no-tracking && python -m churn check --min-roc-auc 0.82


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_MODEL_DIR=/app/models

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY --from=train /app/models ./models

RUN useradd --create-home --uid 10001 api
USER api
EXPOSE 8000

CMD ["uvicorn", "churn.serve:app", "--host", "0.0.0.0", "--port", "8000"]
