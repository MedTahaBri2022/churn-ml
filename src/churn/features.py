"""Feature engineering and preprocessing.

Everything lives inside the scikit-learn pipeline, so the exact same
transformations run at training time and at prediction time: there is no
second implementation to keep in sync in the API.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

ENGINEERED = ["services_count", "average_monthly_spend", "is_new_customer"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features: they use no statistic of the dataset, so they cannot
    leak information from validation folds."""
    df = df.copy()
    df["services_count"] = (df[config.SERVICES] == "Yes").sum(axis=1)
    # What the customer really paid per month so far; the current monthly
    # charge for brand-new customers.
    df["average_monthly_spend"] = (
        df["TotalCharges"].div(df["tenure"].where(df["tenure"] > 0)).fillna(df["MonthlyCharges"])
    )
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    return df


def build_preprocessor() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = OneHotEncoder(
        categories=[config.CATEGORIES[column] for column in config.CATEGORICAL],
        # A category never seen in training becomes all zeros instead of an error.
        handle_unknown="ignore",
        sparse_output=False,
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric, config.NUMERIC + ENGINEERED),
            ("categorical", categorical, config.CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_features, feature_names_out=None)),
            ("columns", columns),
        ]
    )
