"""Model selection, training and artifact export."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from . import config, data, drift, evaluate
from .features import build_preprocessor


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict


def candidates(seed: int) -> list[Candidate]:
    return [
        Candidate(
            "logistic_regression",
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
            ),
            {"model__max_depth": [6, 10], "model__min_samples_leaf": [5, 20]},
        ),
        Candidate(
            "gradient_boosting",
            HistGradientBoostingClassifier(random_state=seed),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5]},
        ),
    ]


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def _git_revision() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def train(
    data_path: Path = config.DATA_PATH,
    model_dir: Path = config.MODEL_DIR,
    report_dir: Path = config.REPORT_DIR,
    seed: int = config.SEED,
    only: list[str] | None = None,
    track: bool = True,
) -> dict:
    """Trains every candidate, keeps the best one and writes the artifacts.

    Selection uses cross-validation on the training split only. The test
    split is used once, after every choice (model, hyper-parameters,
    threshold) has been made.
    """
    df = data.load_dataset(data_path)
    X_train, X_test, y_train, y_test = data.split(df, seed)
    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)

    baseline = cross_val_predict(
        make_pipeline(DummyClassifier(strategy="prior")),
        X_train, y_train, cv=folds, method="predict_proba",
    )[:, 1]
    leaderboard = [
        {"name": "baseline_prior", "params": {}, **evaluate.ranking_metrics(y_train, baseline)}
    ]

    best_search = None
    best_name = ""
    for candidate in candidates(seed):
        if only and candidate.name not in only:
            continue
        search = GridSearchCV(
            make_pipeline(candidate.estimator),
            candidate.grid,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
            refit="roc_auc",
            cv=folds,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        index = search.best_index_
        leaderboard.append(
            {
                "name": candidate.name,
                "params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                "roc_auc": float(search.cv_results_["mean_test_roc_auc"][index]),
                "roc_auc_std": float(search.cv_results_["std_test_roc_auc"][index]),
                "pr_auc": float(search.cv_results_["mean_test_pr_auc"][index]),
            }
        )
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search, best_name = search, candidate.name

    if best_search is None:
        raise ValueError(f"no candidate selected (only={only})")
    model: Pipeline = best_search.best_estimator_

    # Threshold from out-of-fold predictions: the test set stays untouched.
    out_of_fold = cross_val_predict(
        model, X_train, y_train, cv=folds, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = evaluate.choose_threshold(y_train, out_of_fold)

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = {
        **evaluate.ranking_metrics(y_test, test_probabilities),
        **evaluate.decision_metrics(y_test, test_probabilities, threshold),
        "default_threshold": evaluate.decision_metrics(y_test, test_probabilities, 0.5),
    }

    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=-1
    )
    top_features = [
        {"feature": X_test.columns[i], "roc_auc_drop": float(importance.importances_mean[i])}
        for i in np.argsort(importance.importances_mean)[::-1][:8]
    ]

    metadata = {
        "model_name": best_name,
        "params": leaderboard[[row["name"] for row in leaderboard].index(best_name)]["params"],
        "threshold": threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "data_sha256": data.file_sha256(data_path),
        "rows": {"train": int(len(X_train)), "test": int(len(X_test))},
        "churn_rate": float(df[config.TARGET].eq("Yes").mean()),
        "features": config.FEATURES,
        "seed": seed,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    report = {
        "leaderboard_cv": leaderboard,
        "test": test_metrics,
        "top_features": top_features,
        "metadata": metadata,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "reference.json").write_text(
        json.dumps(drift.build_reference(X_train)), encoding="utf-8"
    )
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "model_card.md").write_text(model_card(report), encoding="utf-8")

    if track:
        _log_to_mlflow(report, model_dir, report_dir)
    return report


def _log_to_mlflow(report: dict, model_dir: Path, report_dir: Path) -> None:
    """Experiment tracking: every run is recorded with its parameters,
    metrics and artifacts, so two trainings can be compared later."""
    import mlflow

    # SQLite backend for the runs, a local folder for the artifacts.
    mlflow.set_tracking_uri(f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name("churn") is None:
        mlflow.create_experiment(
            "churn", artifact_location=(config.ROOT / "mlartifacts").as_uri()
        )
    mlflow.set_experiment("churn")
    metadata, test = report["metadata"], report["test"]
    with mlflow.start_run(run_name=metadata["model_name"]):
        mlflow.log_params(
            {"model": metadata["model_name"], "seed": metadata["seed"], **metadata["params"]}
        )
        mlflow.log_param("data_sha256", metadata["data_sha256"][:12])
        for row in report["leaderboard_cv"]:
            mlflow.log_metric(f"cv_roc_auc_{row['name']}", row["roc_auc"])
        for name in ("roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "threshold"):
            mlflow.log_metric(f"test_{name}", test[name])
        mlflow.log_artifact(str(report_dir / "metrics.json"))
        mlflow.log_artifact(str(report_dir / "model_card.md"))
        mlflow.log_artifact(str(model_dir / "model.joblib"))


def model_card(report: dict) -> str:
    metadata, test = report["metadata"], report["test"]
    matrix = test["confusion_matrix"]
    lines = [
        "# Model card: customer churn classifier",
        "",
        f"- **Model**: `{metadata['model_name']}` {metadata['params']}",
        f"- **Trained**: {metadata['trained_at']} (git `{metadata['git_revision']}`)",
        f"- **Data**: {metadata['rows']['train']} training rows, {metadata['rows']['test']} test rows, "
        f"churn rate {metadata['churn_rate']:.1%}, sha256 `{metadata['data_sha256'][:12]}`",
        "",
        "## Model selection (5-fold cross-validation on the training split)",
        "",
        "| Model | ROC-AUC | PR-AUC | Best parameters |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in report["leaderboard_cv"]:
        lines.append(
            f"| {row['name']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['params'] or '-'} |"
        )
    lines += [
        "",
        "## Hold-out test set (used once)",
        "",
        f"- ROC-AUC **{test['roc_auc']:.3f}**, PR-AUC **{test['pr_auc']:.3f}**, Brier score {test['brier']:.3f}",
        f"- Decision threshold **{test['threshold']:.2f}**, chosen on out-of-fold training predictions to "
        f"minimize cost (a missed churner costs {config.COST_FALSE_NEGATIVE:g}x a wasted retention offer)",
        f"- At that threshold: precision {test['precision']:.3f}, recall {test['recall']:.3f}, F1 {test['f1']:.3f}",
        f"- At the default 0.50: precision {test['default_threshold']['precision']:.3f}, "
        f"recall {test['default_threshold']['recall']:.3f}",
        "",
        "| | Predicted stay | Predicted churn |",
        "| --- | ---: | ---: |",
        f"| **Stayed** | {matrix['true_negative']} | {matrix['false_positive']} |",
        f"| **Churned** | {matrix['false_negative']} | {matrix['true_positive']} |",
        "",
        "## Most influential features (permutation importance, drop in ROC-AUC)",
        "",
    ]
    lines += [f"- `{row['feature']}`: {row['roc_auc_drop']:.4f}" for row in report["top_features"]]
    lines += [
        "",
        "## Intended use and limits",
        "",
        "- Ranks customers by churn risk to prioritize retention actions; not a basis for decisions "
        "that affect a customer's rights or pricing.",
        "- Trained on one public snapshot of a telecom operator: no time dimension, so the model "
        "cannot be validated on a later period, which is how it would be used in production.",
        "- `gender` and `SeniorCitizen` are inputs; fairness across those groups has not been audited.",
        "",
    ]
    return "\n".join(lines)


def load_artifacts(model_dir: Path = config.MODEL_DIR):
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    reference = json.loads((model_dir / "reference.json").read_text(encoding="utf-8"))
    return model, metadata, reference


def predict_frame(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[config.FEATURES])[:, 1]
