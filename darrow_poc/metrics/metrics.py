import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_error, recall_score
from sklearn.metrics import make_scorer


def root_mean_squared_error(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))


def mean_absolute_percentage_error(actual, pred):
    mape = np.mean(
        np.abs(
            (actual - pred) / actual
        )
    )
    return mape


def peaks_mae(actual, pred, threshold=60):
    if sum(actual > threshold) == 0:
        return np.nan
    else:
        pred = pred[actual > threshold]
        actual = actual[actual > threshold]
        return mean_absolute_error(actual, pred)


def peaks_mape(actual, pred, threshold=60):
    if sum(actual > threshold) == 0:
        return np.nan
    else:
        pred = pred[actual > threshold]
        actual = actual[actual > threshold]
        return mean_absolute_percentage_error(actual, pred)


def weighted_rmse(actual, pred):
    wrmse = np.sqrt(np.average(
        a=np.power(actual - pred, 2),
        weights=actual
    ))
    return wrmse


def weighted_mae(actual, pred):
    wmae = np.average(
        a=np.abs(actual - pred),
        weights=actual
    )
    return wmae


def peaks_recall(actual, pred, threshold=60):
    actual = np.where(actual > threshold, 1, 0)
    pred = np.where(pred > threshold, 1, 0)
    return recall_score(actual, pred, zero_division=1)


def get_metrics():
    metrics = {
        'rmse': root_mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score,
        'wmae': weighted_mae,
        'peak_mae': peaks_mae,
        'threshold_recall': peaks_recall
    }
    return metrics


def multi_metric(metric):
    def multi(actual, pred):
        return np.mean([metric(a, p) for a, p in zip(actual.T, pred.T)])
    return multi


def get_scorer():

    scorer = {
        'rmse': make_scorer(multi_metric(root_mean_squared_error), greater_is_better=False),
        'mae': make_scorer(multi_metric(mean_absolute_error), greater_is_better=False),
        'r2': make_scorer(multi_metric(r2_score), greater_is_better=True),
        'wmae': make_scorer(multi_metric(weighted_mae), greater_is_better=False),
        'peak_mae': make_scorer(multi_metric(peaks_mae), greater_is_better=False),
        'threshold_recall': make_scorer(multi_metric(peaks_recall), greater_is_better=True)
    }
    return scorer


def calculate_metrics(actual, pred):
    actual = pd.Series(actual)
    pred = pd.Series(pred)
    missing = actual.isna() | pred.isna()
    actual = actual.loc[~missing]
    pred = pred.loc[~missing]
    metrics = get_metrics()
    result = {key: metrics[key](actual, pred) for key in metrics.keys()}
    return result
