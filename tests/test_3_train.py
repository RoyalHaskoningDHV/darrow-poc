import pandas as pd
import numpy as np
import pytest
import dill as pkl

from sklearn.metrics import r2_score

from roer.models.models import (
    train_validator,
    train_imputer,
    train_forecasting_models,
    train_quantile_models,
    assess_model_performances,
)
from roer.models.feature_engineering import (
    engineer_features_for_imputation,
    ValidationTransformer,
)
from roer.preprocessing.abt import create_abt


@pytest.fixture
def train():
    train = pd.read_csv(
        './tests/testing_data/train.csv', parse_dates=['TIME'], index_col=['TIME']
    )
    return train

@pytest.fixture
def train_with_missing_values():
    train_with_missing_values = pd.read_csv(
        './tests/testing_data/train.csv', parse_dates=['TIME'], index_col=['TIME']
    )
    train_with_missing_values.loc[100:200, 'discharge_altenburg1'] = np.nan
    return train_with_missing_values

@pytest.fixture
def test():
    test = pd.read_csv(
        './tests/testing_data/test.csv', parse_dates=['TIME'], index_col=['TIME']
    )
    return test

@pytest.fixture
def test_with_missing_values():
    test_with_missing_values = pd.read_csv(
        './tests/testing_data/test.csv', parse_dates=['TIME'], index_col=['TIME']
    )
    test_with_missing_values.loc[100:200, 'discharge_altenburg1'] = np.nan
    return test_with_missing_values

@pytest.fixture
def trained_validator(train):
    trained_validator, timestamp = train_validator(train, testing=True)
    return trained_validator

@pytest.fixture
def trained_imputer(train):
    trained_imputer, timestamp = train_imputer(train, testing=True)
    return trained_imputer

@pytest.fixture
def full_models(train, trained_validator, trained_imputer):
    full_models = train_forecasting_models(
        train,
        trained_validator,
        trained_imputer,
        testing=True,
    )
    return full_models

def test_train_validator(train, test, train_with_missing_values, test_with_missing_values):
    for trn, tst in [(train, test), (train_with_missing_values, test_with_missing_values)]:
        channel_subset = [
            'discharge_stah',
            'discharge_linnich1',
            'discharge_juelich_wl',
        ]
        validator, timestamp = train_validator(
            trn.loc[:, channel_subset],
            n_features=3,
            testing=True,
        )
        test_val = ValidationTransformer(
            validator=validator
        ).transform(
            tst.loc[:, channel_subset]
        )
        assert test_val.shape == tst.loc[:, channel_subset].shape

def test_train_imputer(test, trained_imputer):
    test_missing = test.copy()
    test_missing.loc[:, 'discharge_stah'] = np.nan

    X = engineer_features_for_imputation(
        test_missing,
        lags_discharge=[1, 2, 3, 4, 5, 8],
        lags_precip=[3]
    )
    X_transform = trained_imputer.transform(X)

    df = test.copy()
    df_transform = pd.DataFrame(X_transform, columns=X.columns, index=X.index)
    non_lag_cols = [c for c in X.columns if 'lag' not in c]
    df.loc[:, non_lag_cols] = df_transform.loc[:, non_lag_cols]

    # since we use a tiny training set, performance might be poor
    assert r2_score(df.loc[:, 'discharge_stah'], test.loc[:, 'discharge_stah']) > 0.25, (
        "The imputation model performs significantly worse than a prior version"
    )

def test_train_forecaster(train, test, full_models):
    df_r2 = assess_model_performances(test, full_models)
    assert df_r2.shape == (24, len(full_models)), (
        "Issue when assessing model performances, output shape suspicious."
    )

def test_train_quantile_models(train, test, full_models):
    quantile_models = train_quantile_models(train, full_models, testing=True)
