"""Loading, cleaning, validating and splitting the dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class DataValidationError(ValueError):
    """Raised with the full list of problems, not just the first one."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path = config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes the known quirks of the raw file without dropping rows.

    `TotalCharges` is text in the source and blank for customers in their
    first month (tenure 0): nothing has been billed yet, so the value is 0,
    not missing.
    """
    df = df.copy()
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    first_month = total.isna() & (df["tenure"] == 0)
    df["TotalCharges"] = total.mask(first_month, 0.0)
    return df


def validate(df: pd.DataFrame, require_target: bool = True) -> None:
    """Schema and range checks. Fails loudly: a model trained on silently
    malformed data is worse than no model."""
    problems: list[str] = []

    expected = list(config.FEATURES) + ([config.TARGET] if require_target else [])
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise DataValidationError([f"missing columns: {', '.join(missing)}"])

    if df.empty:
        problems.append("dataset is empty")

    for column in config.NUMERIC:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            problems.append(f"{column}: {int(values.isna().sum())} non-numeric or missing value(s)")
        elif (values < 0).any():
            problems.append(f"{column}: negative value(s)")

    for column, allowed in config.CATEGORIES.items():
        unknown = sorted(set(df[column].dropna().unique()) - set(allowed), key=str)
        if unknown:
            problems.append(f"{column}: unexpected value(s) {unknown}")
        if df[column].isna().any():
            problems.append(f"{column}: missing value(s)")

    if require_target:
        unknown = sorted(set(df[config.TARGET].dropna().unique()) - {"Yes", "No"})
        if unknown:
            problems.append(f"{config.TARGET}: unexpected value(s) {unknown}")

    if config.ID_COLUMN in df.columns and df[config.ID_COLUMN].duplicated().any():
        problems.append(f"{config.ID_COLUMN}: duplicated identifiers")

    if problems:
        raise DataValidationError(problems)


def load_dataset(path: Path = config.DATA_PATH) -> pd.DataFrame:
    df = clean(load_raw(path))
    validate(df)
    return df


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURES], (df[config.TARGET] == "Yes").astype(int)


def split(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified hold-out: the test set is touched once, at the very end."""
    X, y = features_and_target(df)
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
