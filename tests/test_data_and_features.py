import pandas as pd
import pytest

from churn import config, data
from churn.features import add_features, build_preprocessor
from tests.conftest import frame


def test_dataset_loads_clean(dataset):
    assert len(dataset) == 7043
    assert dataset["TotalCharges"].isna().sum() == 0
    assert 0.25 < dataset[config.TARGET].eq("Yes").mean() < 0.28


def test_blank_total_charges_means_nothing_billed_yet():
    raw = pd.DataFrame({"TotalCharges": [" ", "120.5", " "], "tenure": [0, 4, 3]})
    cleaned = data.clean(raw)

    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 120.5
    # A blank for an existing customer is a real gap, not a zero.
    assert pd.isna(cleaned["TotalCharges"].iloc[2])


def test_validation_reports_every_problem(dataset):
    broken = dataset.head(50).copy()
    broken.loc[broken.index[0], "Contract"] = "Lifetime"
    broken.loc[broken.index[1], "MonthlyCharges"] = -5
    broken.loc[broken.index[2], config.ID_COLUMN] = broken[config.ID_COLUMN].iloc[3]

    with pytest.raises(data.DataValidationError) as raised:
        data.validate(broken)

    assert len(raised.value.problems) == 3
    assert any("Lifetime" in problem for problem in raised.value.problems)


def test_validation_names_missing_columns(dataset):
    with pytest.raises(data.DataValidationError, match="missing columns: tenure"):
        data.validate(dataset.drop(columns=["tenure"]))


def test_split_is_stratified_and_reproducible(dataset):
    _, _, y_train, y_test = data.split(dataset)
    _, _, y_train_again, _ = data.split(dataset)

    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert y_train.index.equals(y_train_again.index)
    assert set(y_train.index).isdisjoint(y_test.index)


def test_engineered_features(customer):
    engineered = add_features(frame(customer)).iloc[0]

    assert engineered["services_count"] == 2
    assert engineered["average_monthly_spend"] == 95.0
    assert engineered["is_new_customer"] == 1


def test_average_spend_of_a_brand_new_customer_is_the_monthly_charge(customer):
    engineered = add_features(frame({**customer, "tenure": 0, "TotalCharges": 0.0})).iloc[0]
    assert engineered["average_monthly_spend"] == 95.0


def test_preprocessor_survives_an_unseen_category(dataset, customer):
    X, _ = data.features_and_target(dataset)
    preprocessor = build_preprocessor().fit(X)

    transformed = preprocessor.transform(frame({**customer, "PaymentMethod": "Crypto"}))

    assert transformed.shape == (1, preprocessor.transform(X.head(1)).shape[1])
    assert not pd.isna(transformed).any()
