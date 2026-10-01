"""Data drift monitoring with the Population Stability Index (PSI).

At training time a compact description of each feature's distribution is
saved next to the model. Later, any batch of production data can be compared
with it without access to the training set.

Rule of thumb for PSI: below 0.1 stable, 0.1 to 0.2 worth watching,
above 0.2 significant shift (retraining should be considered).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

BINS = 10
EPSILON = 1e-4
ALERT_THRESHOLD = 0.2


def build_reference(df: pd.DataFrame) -> dict:
    reference: dict = {"numeric": {}, "categorical": {}}
    for column in config.NUMERIC:
        values = df[column].astype(float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, BINS + 1)))
        inner = edges[1:-1]  # open-ended first and last bins
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        reference["numeric"][column] = {
            "edges": inner.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
        }
    for column in config.CATEGORICAL:
        shares = df[column].astype(str).value_counts(normalize=True)
        reference["categorical"][column] = shares.to_dict()
    return reference


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_psi(reference: dict, df: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column, stats in reference["numeric"].items():
        inner = np.asarray(stats["edges"])
        counts = np.bincount(
            np.searchsorted(inner, df[column].astype(float), side="right"),
            minlength=len(inner) + 1,
        )
        scores[column] = _psi(np.asarray(stats["proportions"]), counts / counts.sum())
    for column, shares in reference["categorical"].items():
        observed = df[column].astype(str).value_counts(normalize=True).to_dict()
        categories = sorted(set(shares) | set(observed))
        scores[column] = _psi(
            np.asarray([shares.get(category, 0.0) for category in categories]),
            np.asarray([observed.get(category, 0.0) for category in categories]),
        )
    return scores


def drifted(scores: dict[str, float], threshold: float = ALERT_THRESHOLD) -> list[str]:
    return sorted(column for column, value in scores.items() if value > threshold)
