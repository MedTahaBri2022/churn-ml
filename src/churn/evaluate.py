"""Metrics and decision-threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def ranking_metrics(y_true, probabilities) -> dict[str, float]:
    """Threshold-free quality: how well the model ranks and how calibrated it is."""
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def expected_cost(y_true, probabilities, threshold: float) -> float:
    predicted = np.asarray(probabilities) >= threshold
    actual = np.asarray(y_true).astype(bool)
    false_negatives = int((actual & ~predicted).sum())
    false_positives = int((~actual & predicted).sum())
    return (
        false_negatives * config.COST_FALSE_NEGATIVE
        + false_positives * config.COST_FALSE_POSITIVE
    ) / len(actual)


def choose_threshold(y_true, probabilities) -> float:
    """Threshold that minimizes the expected cost per customer.

    Called on out-of-fold predictions of the training set only: tuning it on
    the test set would make the reported test metrics optimistic.
    """
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(y_true, probabilities, t) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def decision_metrics(y_true, probabilities, threshold: float) -> dict:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "expected_cost": float(expected_cost(y_true, probabilities, threshold)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
