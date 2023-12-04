from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

from scipy.stats import norm
from sklearn import base
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sam.feature_engineering import BuildRollingFeatures


def engineer_steps(channel: str, channels: list):
    """Prepare feature engineering steps:

    Parameters
    ----------
    channel: str
        Name of channel of interest
    channels : list
        List of all channels in the data

    Returns
    -------
    list
        List of feature engineering steps to perform
    """
    other_discharge_channels = [c for c in channels if (c != channel) & ("discharge" in c)]
    precipitation_channels = [c for c in channels if "precip" in c]

    discharge_step = (
        "lag_features_discharge_channels",
        BuildRollingFeatures(
            rolling_type="lag",
            window_size=[0, 1, 2, 3, 4, 5, 8],
            lookback=0,
            keep_original=False,
        ),
        other_discharge_channels,
    )

    precipitation_step = (
        "lag_features_precipitation_channels",
        BuildRollingFeatures(
            rolling_type="lag",
            window_size=[0, 3],
            lookback=0,
            keep_original=False,
        ),
        precipitation_channels,
    )

    if len(precipitation_channels) > 0:
        return [discharge_step, precipitation_step]

    return [discharge_step]


def get_outliers(
    y_true,
    y_hat,
    outlier_min_q: int = 3,
    outlier_window: int = 1,
    outlier_limit: int = 1,
):
    """Determine outliers, similar to `sam_quantile_plot` implementation.

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
    n_consensus: Union[int, str] = "all",
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
    n_consensus: Union[int, str] (default = 'all')
        By default all sub-model predictions have to flag outliers, but you can also specify
        an integer of the number of models desired for consenus.
        TODO: Currently only 'all' is implemented.
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
    n_consensus: Union[int, str] = "all",
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
    n_consensus: Union[int, str] (default = 'all')
        By default all sub-model predictions have to flag outliers, but you can also specify
        an integer of the number of models desired for consenus.
        TODO: Currently only 'all' is implemented.
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
        outlier_window=outlier_window,
        outlier_limit=outlier_limit,
    )

    return anomalies


def standardize_prediction_column_names(y_hat):
    return y_hat.rename(
        columns={
            "predict_q_0.9986501019683699": "predict_lead_0_q_0.9986501019683699",
            "predict_q_0.9772498680518208": "predict_lead_0_q_0.9772498680518208",
            "predict_q_0.8413447460685429": "predict_lead_0_q_0.8413447460685429",
            "predict_q_0.15865525393145707": "predict_lead_0_q_0.15865525393145707",
            "predict_q_0.02275013194817921": "predict_lead_0_q_0.02275013194817921",
            "predict_q_0.0013498980316301035": "predict_lead_0_q_0.0013498980316301035",
            "predict_q_0.5": "predict_lead_0_mean",  # Note that we actually have the median here
        }
    )


class ValidationModel(base.BaseEstimator, base.RegressorMixin):
    """Convenience class for building and running a MLPRegression model for anomaly
    detection. For a given channel, we take the top features and use all combinations of
    leaving one feature out to train the models. This way we can ensure we find true
    anomalies in the target channel rather than anomalies depending on one of the features.
    We accept anomalies that are flagged by all (or most) of those models.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with channels as column names
    model_type : str (default='lasso')
        Could also be 'mlp'.
    training_end_date : str (default = '2021-07-31 23:59:59')
        Marks the endpoint of the training data and start of the test data
    epochs : int (default = 2)
        Number of epochs for MLP
    n_features : int (default = 5)
        Number of features to include in models + 1. So if you want to have 3 features
        in each sub-model this value should be 4.
    n_consensus: Union[int, str] (default = 'all')
        By default all sub-model predictions have to flag outliers, but you can also specify
        an integer of the number of models desired for consenus.
        TODO: Currently only 'all' is implemented.
    outlier_window: int (default=1)
        the window size in which at least `outlier_limit` should be outside of `outlier_min_q`
    outlier_limit: int (default=1)
        the minimum number of outliers within outlier_window to be outside of `outlier_min_q`
    use_precipitation_features: bool (default=False)
        Whether or not to use precipitation features in models. If True, we will use all
        precipitation features in all models with lags of [0, 3]
    learning_rate : float (default=0.001)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_type: str = "lasso",
        training_end_date: str = None,
        epochs: int = 2,
        n_features: int = 5,
        n_consensus: Union[int, str] = "all",
        outlier_window: int = 3,
        outlier_limit: int = 3,
        use_precipitation_features: bool = False,
        learning_rate: float = 0.001,
    ):
        self.df = df.copy()
        self.model_type = model_type
        self.discharge_channels = [c for c in self.df.columns if "discharge" in c]
        self.training_end_date = self._get_training_end_date(training_end_date)
        self.epochs = epochs
        self.n_features = n_features
        self.n_consensus = n_consensus
        self.outlier_window = outlier_window
        self.outlier_limit = outlier_limit
        self.use_precipitation_features = use_precipitation_features
        self.learning_rate = learning_rate

    def _get_training_end_date(self, training_end_date: str = None):
        """We can either provide a dataframe with both training and test set, in
        which case the parameter `training_end_date` denotes the cut-off; or we can
        provide only a training set, in which case training_end_date should be None.
        Then this method returns the last date of the input data.

        Parameters
        ----------
        training_end_date: str (default=None)

        Returns
        -------
        str
            The end date of the training data (input data on initialization)
            if `training_end_date` is None, otherwise `training_end_date`.
        """
        if training_end_date is None:
            return self.df.index.max()
        return training_end_date

    def _train_test_split(self, channel: str, training_end_date: str = "2017-12-31 23:59:59"):
        """Split data into train and test sets based on datetime cutoff.
        Everything before the cutoff is training data, everything after is
        test data.

        Parameters
        ----------
        channel: str
            Name of channel of interest (target variable)
        training_end_date: str (default = '2017-12-31 23:59:59')

        Returns
        -------
        X_train : pd.DataFrame
        y_train : pd.Series
        X_test : pd.DataFrame
        y_test : pd.Series
        """
        X_train = self.df.loc[:training_end_date, :].copy().reset_index(drop=True)
        y_train = self.df.loc[:training_end_date, channel].copy().reset_index(drop=True)
        X_test = self.df.loc[training_end_date:, :].copy().reset_index(drop=True)
        y_test = self.df.loc[training_end_date:, channel].copy().reset_index(drop=True)
        self.n_samples_train = len(y_train)
        self.n_samples_test = len(y_test)

        return X_train, y_train, X_test, y_test

    def _get_feature_channels(self, channel: str):
        """Get MLP model for particular channel

        Parameters
        ----------
        channel: str
            Name of channel of interest

        Returns
        -------
        feature_channels: list
            List of strings denoting the channels to use for feature engineering when
            predicting `channel`
        """
        discharge = self.df.loc[: self.training_end_date, self.discharge_channels]

        correlations = discharge.corr("spearman").loc[channel]
        sort_ids = np.argsort(correlations.values)
        feature_channels = correlations.index[sort_ids][-1 - self.n_features : -1]

        return feature_channels

    def _get_pipeline(
        self,
        channel: str,
        feature_channels: list,
    ):
        """Prepare ML pipeline

        Parameters
        ----------
        channel: str
            Name of channel of interest
        feature_channels: list
            List of channel names to use for feature engineering
        """
        engineer = ColumnTransformer(engineer_steps(channel, feature_channels), remainder="drop")
        scaler = StandardScaler()
        pipe = Pipeline(
            [
                ("columns", engineer),
                ("scaler", scaler),
                ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ]
        )
        return pipe

    def _get_mlp_model(self, channel: str, feature_channels: list):
        """Get MLP model for particular channel

        Parameters
        ----------
        channel: str
            Name of channel of interest
        feature_channels: list
            List of channel names to use for feature engineering

        Returns
        -------
        estimator : sklearn Model
        """
        from sam.models import MLPTimeseriesRegressor

        self.quantiles = (
            norm.cdf(3),
            norm.cdf(2),
            norm.cdf(1),
            1 - norm.cdf(1),
            1 - norm.cdf(2),
            1 - norm.cdf(3),
        )
        self.predict_ahead = (0,)

        estimator = MLPTimeseriesRegressor(
            predict_ahead=self.predict_ahead,
            quantiles=self.quantiles,
            feature_engineer=self._get_pipeline(channel, feature_channels),
            epochs=self.epochs,
            verbose=0,
            average_type="median",
            lr=self.learning_rate,
        )

        return estimator

    def _get_lasso_model(self, channel: str, feature_channels: list):
        """Get MLP model for particular channel

        Parameters
        ----------
        channel: str
            Name of channel of interest
        feature_channels: list
            List of channel names to use for feature engineering

        Returns
        -------
        estimator : sklearn Model
        """
        from darrow_poc.models.quantreg import LinearQuantileRegressor

        regressor = LinearQuantileRegressor(
            quantiles=[
                norm.cdf(3),
                norm.cdf(2),
                norm.cdf(1),
                norm.cdf(0),
                1 - norm.cdf(1),
                1 - norm.cdf(2),
                1 - norm.cdf(3),
            ],
        )

        estimator = Pipeline(
            [
                ("preprocessor", self._get_pipeline(channel, feature_channels)),
                ("regressor", regressor),
            ]
        )

        return estimator

    def _get_model(self, channel: str, feature_channels: list):
        """Get MLP model for particular channel

        Parameters
        ----------
        channel: str
            Name of channel of interest
        feature_channels: list
            List of channel names to use for feature engineering

        Returns
        -------
        estimator : sklearn Model
        """
        if self.model_type == "mlp":
            return self._get_mlp_model(channel, feature_channels)
        elif self.model_type == "lasso":
            return self._get_lasso_model(channel, feature_channels)
        else:
            raise NotImplementedError(
                f"model_type {self.model_type} not implemented." "Choose `mlp` or `lasso` instead."
            )

    def fit_and_evaluate(self, target_channel: str):
        """Fit ML model for each channel and evaluate performance on test set.
        
        Parameters
        ----------
        target_channel: str
            Name of target channel

        Returns
        -------
        model : dict
            Model object for each channel
        num_obs : dict
            Numbers of observations in training and test data
        pred : dict
            Predictions for each channel for test set
        r2 : pd.DataFrame
            Contains r2 scores
        """
        model, num_obs, pred, r2 = {}, {}, {}, {}
        model[target_channel], pred[target_channel], r2[target_channel] = {}, {}, {}

        # Get training and test data
        X_train, y_train, X_test, y_test = self._train_test_split(target_channel, self.training_end_date)

        num_train = np.sum(~y_train.isna())
        num_test = np.sum(~y_test.isna())
        num_obs[f"train_{target_channel}"] = num_train
        num_obs[f"test_{target_channel}"] = num_test

        # Determine which channels to use for feature engineering
        feature_channels = self._get_feature_channels(target_channel)

        for leave_out_feature in feature_channels:
            feature_channel_subset = [c for c in feature_channels if c != leave_out_feature]

            print(
                f"\nTraining model for channel {target_channel} in time period from "
                f"{X_train.index[0]} to {self.training_end_date}."
                f"\n We use the following feature channels: {feature_channel_subset}"
            )

            # Fit model
            model[target_channel][leave_out_feature] = self._get_model(target_channel, feature_channel_subset)
            model[target_channel][leave_out_feature].fit(X_train, y_train)

            if hasattr(model[target_channel][leave_out_feature], "model_"):
                from tensorflow.keras.optimizers import Adam

                model[target_channel][leave_out_feature].model_.compile(  # HACK
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss=None,
                )

            # Evaluate
            pred[target_channel][leave_out_feature] = standardize_prediction_column_names(
                model[target_channel][leave_out_feature].predict(X_test)
            )

            if isinstance(pred[target_channel][leave_out_feature], pd.DataFrame):
                y_hat = pred[target_channel][leave_out_feature].loc[:, "predict_lead_0_mean"]
            else:
                y_hat = pred[target_channel][leave_out_feature]
            finite_selection = ~y_test.isna() & ~np.isnan(y_hat)
            if finite_selection.sum() > 0:
                r2[target_channel][leave_out_feature] = r2_score(y_test[finite_selection], y_hat[finite_selection])
            else:
                r2[target_channel][leave_out_feature] = np.nan

        self.model = model
        self.num_obs = num_obs
        self.pred = pred
        self.r2 = r2

        return model, num_obs, pred, r2

    def flatten_output(self, output: dict[str, dict], name: str) -> dict[str, str]:
        return {
            f"{name}_target_{outer_key}_missing_feature_{inner_key}": f"{inner_value}"
            for outer_key, inner_dict in output.items()
            for inner_key, inner_value in inner_dict.items()
        }

    def predict(
        self,
        X: pd.DataFrame,
        target_channel,
    ):
        """Predict for each target channel and left out feature channel.

        Returns
        -------
        pred : dict
            Predictions for each channel for test set
        r2 : pd.DataFrame
            Contains r2 scores
        """
        pred = {}
        pred[target_channel] = {}
        X_test = X.loc[:, [c for c in X.columns if c != target_channel]].reset_index(drop=True)

        feature_channels = self._get_feature_channels(target_channel)

        for leave_out_feature in feature_channels:

            pred[target_channel][leave_out_feature] = standardize_prediction_column_names(
                self.model[target_channel][leave_out_feature].predict(X_test)
            )

        return pred
