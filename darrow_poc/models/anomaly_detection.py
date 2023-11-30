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
    anomalies : dict
        The outliers for each target channel.
    """
    anomalies = {}
    for target_channel in [c for c in df_test.columns if "discharge" in c]:
        y_true = df_test.loc[:, target_channel]
        outliers = get_outlier_consensus(
            y_true,
            pred,
            target_channel,
            outlier_window=outlier_window,
            outlier_limit=outlier_limit,
        )
        anomalies[target_channel] = outliers

    return anomalies


def replace_anomalies_with_nan(anomalies: dict, df: pd.DataFrame):
    """Replace all anomalies described in `anomalies` with np.nan in
    `df`.

    Parameters
    ----------
    anomalies : dict
        Dictionary describing where and when anomalies were identified
    df : pd.DataFrame
        Input data

    Returns
    -------
    df_clean : pd.DataFrame
        Same as input `df`, but with np.nan where anomalies were
    """
    df = df.copy()
    for channel, anomaly_array in anomalies.items():
        print(f"replacing {np.sum(anomaly_array)} anomalies with np.nan, channel {channel}")
        df.loc[:, channel].iloc[anomaly_array] = np.nan

    return df


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

    def show_train_test_split(self, training_end_date=None):
        """Convenience method for visualizing training and test set"""
        if training_end_date is None:
            training_end_date = self.training_end_date

        plt.plot(self.df.loc[:training_end_date, :], color="b")
        plt.plot(self.df.loc[training_end_date:, :], color="r")

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

    def _combine_results(self, num_obs: dict, r2: dict):
        """Combine number of observations used for evaluation and fitting and
        r2 scores into convenience dataframe

        Parameters
        ----------
        num_obs : dict
            Contains number of observations for training and testing per channel
        r2 : dict
            Contains r2 scores per channel based on test set

        Returns
        -------
        results : pd.DataFrame
            Contains number of observations for train and test + r2 scores
        """
        results = pd.DataFrame(
            {
                "num_test_data": [num_obs[f"test_{channel}"] for channel in self.discharge_channels],
                "num_train_data": [num_obs[f"train_{channel}"] for channel in self.discharge_channels],
                "r2": [v for v in r2.values()],
            }
        )
        results.index = self.discharge_channels

        return results

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit ML model for each discharge channel

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        model : dict
            Model object for each channel
        """
        model = {}

        for target_channel in self.discharge_channels:
            model[target_channel] = {}

            X_train = X.copy().reset_index(drop=True)
            y_train = X.loc[:, target_channel].copy().reset_index(drop=True)

            # Determine which channels to use for feature engineering
            # TODO: Technically, we shoudl only use the train set for this
            feature_channels = self._get_feature_channels(target_channel)

            for leave_out_feature in feature_channels:
                feature_channel_subset = [c for c in feature_channels if c != leave_out_feature]

                print(
                    f"\nTraining model for channel {target_channel}."
                    f"\n We use the following feature channels: {feature_channel_subset}"
                )

                # Fit model
                model[target_channel][leave_out_feature] = self._get_model(target_channel, feature_channel_subset)
                model[target_channel][leave_out_feature].fit(X_train, y_train)

                # HACK: Remove loss function, because we cannot pickle it
                # It is not needed for makeing predictions, so this is somewhat ok,
                # Otherwise we have to reconstruct it.
                if hasattr(model[target_channel][leave_out_feature], "model_"):
                    from tensorflow.keras.optimizers import Adam

                    model[target_channel][leave_out_feature].model_.compile(
                        optimizer=Adam(learning_rate=self.learning_rate),
                        loss=None,
                    )

        self.model = model

        return self

    def fit_and_evaluate(self):
        """Fit ML model for each channel and evaluate performance on test set.

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

        for target_channel in self.discharge_channels:
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

                print("Coefficient of determination (variance explained) = " f"{r2[target_channel][leave_out_feature]}")

        self.model = model
        self.num_obs = num_obs
        self.pred = pred
        self.r2 = r2

        return model, num_obs, pred, r2

    def predict_all(
        self,
        X: pd.DataFrame,
    ):
        """Predict for each target channel and left out feature channel.

        Returns
        -------
        pred : dict
            Predictions for each channel for test set
        r2 : pd.DataFrame
            Contains r2 scores
        """
        pred, r2 = {}, {}
        for target_channel in self.discharge_channels:
            pred[target_channel], r2[target_channel] = {}, {}
            X_test = X.loc[:, [c for c in X.columns if c != target_channel]].reset_index(drop=True)
            y_test = X.loc[:, target_channel].reset_index(drop=True)

            feature_channels = self._get_feature_channels(target_channel)

            for leave_out_feature in feature_channels:
                feature_channel_subset = [c for c in feature_channels if c != leave_out_feature]

                pred[target_channel][leave_out_feature] = standardize_prediction_column_names(
                    self.model[target_channel][leave_out_feature].predict(X_test)
                )

                if type(pred[target_channel][leave_out_feature]) == pd.DataFrame:
                    y_hat = pred[target_channel][leave_out_feature].loc[:, "predict_lead_0_mean"]
                else:
                    y_hat = pred[target_channel][leave_out_feature]
                finite_selection = ~y_test.isna() & ~np.isnan(y_hat)
                if finite_selection.sum() > 0:
                    r2[target_channel][leave_out_feature] = r2_score(y_test[finite_selection], y_hat[finite_selection])
                else:
                    r2[target_channel][leave_out_feature] = np.nan

                print(
                    f"\nPredicting for channel {target_channel} in time period from "
                    f"{X_test.index[0]} to {X_test.index[-1]}."
                    f"\n We use the following feature channels: {feature_channel_subset}"
                )
                print("Coefficient of determination (variance explained) = " f"{r2[target_channel][leave_out_feature]}")

        return pred, r2

    def predict(
        self,
        X: pd.DataFrame,
    ):
        """Predict for each channel based on test_data.

        Returns
        -------
        pred : dict
            Predictions for each channel for test set
        r2 : pd.DataFrame
            Contains r2 scores
        """
        pred, _ = self.predict_all(X)
        anomalies = get_anomalies(
            pred,
            X,
            n_consensus=self.n_consensus,
            outlier_window=self.outlier_window,
            outlier_limit=self.outlier_limit,
        )
        return replace_anomalies_with_nan(anomalies, X)

    def plot_model_performance(
        self,
        pred: dict,
        results: pd.DataFrame,
        test_data: pd.DataFrame = None,
        training_end_date: str = None,
    ):
        """Make correlation scatter plots of expected results and predicted
        results for each channel.

        Parameters
        ----------
        pred : dict
            Predictions for each channel for test set
        results : pd.DataFrame
            Contains number of observations for train and test + r2 scores
        test_data : pd.DataFrame (default=None)
            For each channel we select y_test based on test_data and training_end_date.
            If test_data is not provided, we use the input data to the etf_model class.
        training_end_date : str (default=None)
            When provided use this as cutoff for train/test data, otherwise use
            self.training_end_date.
        """
        if training_end_date is None:
            training_end_date = self.training_end_date

        fig, axs = plt.subplots(4, 4)
        fig.tight_layout()
        fig.set_figwidth(15)
        fig.set_figheight(12)

        channels = results.index
        for i, channel in enumerate(channels):
            j = i // 4
            ip = i % 4

            if test_data is None:
                y_test = self.df.loc[training_end_date:, channel].copy()
            else:
                y_test = test_data.loc[:, channel].copy()

            if type(pred[channel]) == pd.DataFrame:
                y_hat = pred[channel].loc[:, "predict_lead_0_mean"]
            else:
                y_hat = pred[channel]
            finite_selection = ~y_test.isna() & ~np.isnan(y_hat)

            df_plot = pd.DataFrame(
                {"True values": y_test.loc[finite_selection], "Predictions": y_hat[finite_selection]}
            )
            sns.regplot(ax=axs[ip, j], data=df_plot, x="True values", y="Predictions", ci=True)
            axs[ip, j].set_xlabel("True values")
            axs[ip, j].set_ylabel("Predictions")
            axs[ip, j].set_title(f'{channel}; R2={results["r2"].iloc[i]: .2f}', size=10)


def train_validator(
    df: pd.DataFrame,
    model_type: str = "lasso",
    n_features: int = 5,
    testing: bool = False,
    use_precipitation_features: bool = False,
):
    """train validation / anomaly detection model, with specified lags
    and features. Write result to file.

    Parameters
    ----------
    df : pd.DataFrame
        training data
    model_type : str (default='lasso')
        Which model type to use for anomaly detection ('lasso' or 'mlp')
    n_features : int (default = 5)
        Number of features to include in models + 1. So if you want to have 3 features
        in each sub-model this value should be 4.
    testing : bool (default=False)
        Only set to True when running tests, ommits writing data to file,
        returns model instead.
    use_precipiation_features : bool (default=False)
        Whether or not to use precipitation features for anomaly detection (lags 0, 3)

    Returns
    -------
    imputer : sklearn.Pipeline
        Pipeline of trained validation model.
    timestamp : str
        Returning the timestamp makes it easy to immediately continue with this
        model, also wenn loading it from file.
    """
    validator = ValidationModel(
        df,
        model_type=model_type,
        n_features=n_features,
        use_precipitation_features=use_precipitation_features,
    )

    # We fit many models, but sklearn needs one y input, fake it!
    fake_y = df.iloc[:, 0]
    _ = validator.fit(df, fake_y)

    timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")

    return validator, timestamp
