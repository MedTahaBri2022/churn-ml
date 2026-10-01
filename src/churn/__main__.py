"""Command line.

    python -m churn train                    # select, train, export, track
    python -m churn check --min-roc-auc 0.82 # quality gate for CI
    python -m churn drift new_customers.csv  # compare a batch with training data
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config, data, drift
from .train import load_artifacts, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churn")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and export the model")
    train_parser.add_argument("--only", nargs="*", help="restrict to these candidate models")
    train_parser.add_argument("--no-tracking", action="store_true", help="skip MLflow logging")

    check_parser = commands.add_parser("check", help="fail if the model is below the bar")
    check_parser.add_argument("--min-roc-auc", type=float, default=0.82)

    drift_parser = commands.add_parser("drift", help="PSI of a new batch against training data")
    drift_parser.add_argument("csv")

    args = parser.parse_args(argv)

    if args.command == "train":
        report = train(only=args.only, track=not args.no_tracking)
        for row in report["leaderboard_cv"]:
            print(f"cv  {row['name']:22} roc_auc={row['roc_auc']:.4f} pr_auc={row['pr_auc']:.4f}")
        test = report["test"]
        print(
            f"selected {report['metadata']['model_name']}: test roc_auc={test['roc_auc']:.4f} "
            f"pr_auc={test['pr_auc']:.4f} threshold={test['threshold']:.2f} "
            f"precision={test['precision']:.3f} recall={test['recall']:.3f}"
        )
        return 0

    if args.command == "check":
        report = json.loads((config.REPORT_DIR / "metrics.json").read_text(encoding="utf-8"))
        roc_auc = report["test"]["roc_auc"]
        baseline = report["leaderboard_cv"][0]["roc_auc"]
        print(f"test roc_auc={roc_auc:.4f} (minimum {args.min_roc_auc}, baseline {baseline:.2f})")
        return 0 if roc_auc >= args.min_roc_auc else 1

    if args.command == "drift":
        _, _, reference = load_artifacts()
        batch = data.clean(pd.read_csv(args.csv))
        data.validate(batch, require_target=False)
        scores = drift.compute_psi(reference, batch)
        for column, value in sorted(scores.items(), key=lambda item: -item[1]):
            flag = "DRIFT" if value > drift.ALERT_THRESHOLD else ""
            print(f"{column:18} psi={value:.4f} {flag}")
        shifted = drift.drifted(scores)
        print(f"{len(shifted)} feature(s) above {drift.ALERT_THRESHOLD}: {', '.join(shifted) or 'none'}")
        return 1 if shifted else 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
