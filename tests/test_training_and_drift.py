import json

import numpy as np

from churn import data, drift, evaluate
from churn.train import load_artifacts, train


def test_model_clearly_beats_the_baseline(trained):
    report = trained["report"]
    baseline, model = report["leaderboard_cv"][0], report["leaderboard_cv"][1]

    assert baseline["name"] == "baseline_prior"
    assert abs(baseline["roc_auc"] - 0.5) < 0.01
    assert model["roc_auc"] > 0.80
    assert report["test"]["roc_auc"] > 0.80


def test_training_is_reproducible(trained, tmp_path):
    again = train(
        model_dir=tmp_path / "m", report_dir=tmp_path / "r",
        only=["logistic_regression"], track=False,
    )
    assert again["test"]["roc_auc"] == trained["report"]["test"]["roc_auc"]
    assert again["metadata"]["threshold"] == trained["report"]["metadata"]["threshold"]


def test_artifacts_are_complete(trained):
    model, metadata, reference = load_artifacts(trained["model_dir"])

    assert hasattr(model, "predict_proba")
    assert 0 < metadata["threshold"] < 1
    assert len(metadata["data_sha256"]) == 64
    assert set(reference) == {"numeric", "categorical"}
    report = json.loads((trained["report_dir"] / "metrics.json").read_text(encoding="utf-8"))
    assert report["test"]["confusion_matrix"]["true_positive"] > 0
    assert (trained["report_dir"] / "model_card.md").read_text(encoding="utf-8").startswith("# Model card")


def test_cost_based_threshold_favours_recall(trained):
    test = trained["report"]["test"]

    # Missing a churner is the expensive error, so the chosen threshold is
    # below 0.5 for an uncalibrated-cost view and catches more churners.
    assert test["recall"] >= test["default_threshold"]["recall"]
    assert test["expected_cost"] <= test["default_threshold"]["expected_cost"]


def test_choose_threshold_minimizes_cost():
    y = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.6, 0.55, 0.9])

    threshold = evaluate.choose_threshold(y, probabilities)

    best = evaluate.expected_cost(y, probabilities, threshold)
    assert all(best <= evaluate.expected_cost(y, probabilities, t) for t in (0.3, 0.5, 0.7, 0.95))


def test_no_drift_on_the_training_distribution(dataset, trained):
    _, _, reference = load_artifacts(trained["model_dir"])
    X_train, X_test, _, _ = data.split(dataset)

    assert max(drift.compute_psi(reference, X_train).values()) < 1e-6
    assert drift.drifted(drift.compute_psi(reference, X_test)) == []


def test_drift_is_detected_on_a_shifted_population(dataset, trained):
    _, _, reference = load_artifacts(trained["model_dir"])
    X, _ = data.features_and_target(dataset)

    # A campaign that only brings month-to-month customers paying much more.
    shifted = X[X["Contract"] == "Month-to-month"].copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 1.8

    flagged = drift.drifted(drift.compute_psi(reference, shifted))
    assert "Contract" in flagged
    assert "MonthlyCharges" in flagged
