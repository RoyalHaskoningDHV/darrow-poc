from typing import Literal
import pandas as pd
import numpy as np


def get_outliers(
    y_true,
    y_hat,
    outlier_min_q: int = 3,
    outlier_window: int = 1,
    outlier_limit: int = 1,
):
    """Determine outliers, similar to `sam_quantile_plot` implementation of SAM. Methods referenced
    in the parameter descriptionsn also refer to SAM code.

    See: https://github.com/RoyalHaskoningDHV/sam

    Parameters
    ----------
    y_true: pd.Series
        Pandas Series containing the actual values. Should have same index as y_hat.
    y_hat: pd.DataFrame
        Dataframe returned by the MLPTimeseriesRegressor .predict() function.
        Columns should contain at least `predict_lead_x_mean`, where x is predict ahead
        and for each quantile: `predict_lead_x_q_y` where x is the predict_ahead, and
        y is the quantile. So e.g.:
        `['predict_lead_0_q_0.25, predict_lead_0_q_0.75, predict_lead_mean']`
    outlier_window: int (default=1)
        the window size in which at least `outlier_limit` should be outside of `outlier_min_q`
    outlier_limit: int (default=1)
        the minimum number of outliers within outlier_window to be outside of `outlier_min_q`

    Returns
    -------
    outliers : np.ndarray
        Array with true false values denoting outliers with true
    """
    if isinstance(y_true, pd.core.series.Series):
        y_true = y_true.values

    predict_ahead = 0
    these_cols = [c for c in y_hat.columns if "predict_lead_%d_q_" % predict_ahead in c]
    col_order = np.argsort([float(c.split("_")[-1]) for c in these_cols])
    n_quants = int((len(these_cols)) / 2)

    valid_low = y_hat[these_cols[col_order[n_quants - 1 - (outlier_min_q - 1)]]]
    valid_high = y_hat[these_cols[col_order[n_quants + (outlier_min_q - 1)]]]
    outliers = (y_true > valid_high) | (y_true < valid_low)
    outliers = outliers.astype(int)
    k = np.ones(outlier_window)
    outliers = (np.convolve(outliers, k, mode="full")[: len(outliers)] >= outlier_limit).astype(bool)

    return outliers


def get_outlier_consensus(
    y_true: pd.Series,
    pred: pd.DataFrame,
    target_channel: str,
    n_consensus: int | Literal["all"],
    outlier_min_q: int = 3,
    outlier_window: int = 1,
    outlier_limit: int = 1,
):
    """Determine outliers. We only consider values to be outliers when they occur in
    all or most sub-model predictions. For instance, we might fit 4 models for the target channel,
    where each time we leave one feature out. Then we consider those values outliers that
    are flagged by all or most sub-models.

    Parameters
    ----------
    y_true: pd.Series
        Pandas Series containing the actual values. Should have same index as y_hat.
    y_hat: pd.DataFrame
        Dataframe returned by the MLPTimeseriesRegressor .predict() function.
        Columns should contain at least `predict_lead_x_mean`, where x is predict ahead
        and for each quantile: `predict_lead_x_q_y` where x is the predict_ahead, and
        y is the quantile. So e.g.:
        `['predict_lead_0_q_0.25, predict_lead_0_q_0.75, predict_lead_mean']`
    target_channel: str
        Name of target channel to make predictions for
    n_consensus: int | Literal["all"], (default = 'all')
        By default all sub-model predictions have to flag outliers, but you can also specify
        an integer of the number of models desired for consenus.
    outlier_window: int (default=1)
        the window size in which at least `outlier_limit` should be outside of `outlier_min_q`
    outlier_limit: int (default=1)
        the minimum number of outliers within outlier_window to be outside of `outlier_min_q`

    Returns
    -------
    outliers : np.ndarray
        Array with true false values denoting outliers with true
    """
    outliers = []
    for left_out_channel in pred[target_channel].keys():
        y_hat = pred[target_channel][left_out_channel]
        outliers.append(
            get_outliers(
                y_true,
                y_hat,
                outlier_min_q=outlier_min_q,
                outlier_window=outlier_window,
                outlier_limit=outlier_limit,
            )
        )

    if n_consensus == "all":
        return np.array(outliers).all(axis=0)
    return np.array(outliers).sum(axis=0) >= n_consensus


def get_anomalies(
    pred: dict,
    df_test: pd.DataFrame,
    target_channel: str,
    n_consensus: int | Literal["all"] = "all",
    outlier_window: int = 3,
    outlier_limit: int = 3,
):
    """Get outliers for all features in df_test

    Parameters
    ----------
    pred : dict
        The `pred` key in the output dictionary from the `predict` method of the
        `ValidationModel` class.
        It contains predictions for each target channel for sub-models with single
        channels left out (pred[<target_channel>][<left_out_channel>])
    df_test : pd.DataFrame
        Data where to find anomalies
    target_channel: str
    n_consensus: int | Literal["all"], (default = 'all')
        By default all sub-model predictions have to flag outliers, but you can also specify
        an integer of the number of models desired for consenus.
    outlier_window: int (default=1)
        the window size in which at least `outlier_limit` should be outside of `outlier_min_q`
    outlier_limit: int (default=1)
        the minimum number of outliers within outlier_window to be outside of `outlier_min_q`

    Returns
    -------
    anomalies : np.ndarray
        Array with true false values denoting outliers with true
    """
    y_true = df_test.loc[:, target_channel]
    anomalies = get_outlier_consensus(
        y_true,
        pred,
        target_channel,
        n_consensus=n_consensus,
        outlier_window=outlier_window,
        outlier_limit=outlier_limit,
    )

    return anomalies
