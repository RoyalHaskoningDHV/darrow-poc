import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast
import warnings

import pytz
import pandas as pd
import numpy as np

from sklearn import base
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

logger = logging.getLogger(__name__)

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def create_target(data, horizons, differencing=False):
    y = pd.DataFrame()
    for horizon in horizons:
        y[f'discharge_Stah_lead_{horizon}'] = data['discharge_stah:disc'].shift(-horizon)
    if differencing:
        y = y.sub(data['discharge_stah:disc'], axis=0)
    return y

def build_lag_features(s, lags=[12, 24]):
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features

    Parameters
    ----------
    s : pd.DataFrame or pd.Series
    lags : list
        Lags in samples to add to existing columns in s

    Returns
    -------
    res : pd.DataFrame
        Same as input s, but with lag columns added.

    from
    https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
    '''
    if type(s) is pd.DataFrame:
        new_dict = {}
        for col_name in s:
            new_dict[col_name] = s[col_name]
            # create lagged Series
            for l in lags:
                new_dict['{}_lag{}'.format(col_name, l)] = s[col_name].shift(l).bfill().ffill()
        res = pd.DataFrame(new_dict, index=s.index)

    elif type(s) is pd.Series:
        res = pd.concat([s.shift(l).bfill() for l in lags], axis=1)
        res.columns = ['{}_lag{}'.format(s.name, l) for l in lags]
    else:
        raise NotImplementedError('Only works for DataFrame or Series')
        return None

    return res

def engineer_features_for_imputation(
    df,
    lags_discharge=[1, 2, 3, 4, 5, 8],
    lags_precip=[3],
):
    """From `abt` data (output of create_abt()), add lags for all discharge
    columns, and for one precipitation column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    lags_discharge : list (default=[1, 2, 3, 4, 5, 8])
        Lagged features to create for discharge sensors
    lags_precip : list (default=[3])
        Lagged features to create for worm and upper roer precip.

    Returns
    -------
    features
    """
    discharge_columns = [c for c in df.columns if 'discharge' in c]
    precip_columns = [c for c in df.columns if 'precip' in c]
    precip_column_selection = ['precip_benedenroer:prec', 'precip_worm:prec']

    df_discharge = build_lag_features(df.loc[:, discharge_columns], lags=lags_discharge)
    df_precip = build_lag_features(df.loc[:, precip_column_selection], lags=lags_precip)

    return pd.concat([df_discharge, df_precip], axis=1)

def add_precipitation_forecast_features(
    df_features: pd.DataFrame,
    method=None,
) -> pd.DataFrame:
    """Add precipitation forecast features to feature dataframe.

    Parameters
    ----------
    df_features : pd.DataFrame
        Dataframe with features we also want to use
    method : str (default=None)
        Specifies which type of precipitation forecast to use for training.
        Options:
            1. 'true_rainfall' --> uses actual historical precipitation
            2. 'true_rainfall_plus_noise' --> uses historical precipitation + noise
            3. 'forecast_open_meteo' --> uses open meteo forecast
            4. 'forecast_knmi' --> uses the harmonie40 forecast from the KNMI
            5. None --> Nothing will be added to the input dataframe

    Returns
    -------
    pd.DataFrame
    """
    if method == 'true_rainfall':
        from ..data_sources.precip_forecast import get_open_meteo_historic_rainfall
        df_forecast = get_open_meteo_historic_rainfall()
    elif method == 'true_rainfall_plus_noise':
        from ..data_sources.precip_forecast import get_open_meteo_historic_rainfall_plus_noise
        df_forecast = get_open_meteo_historic_rainfall_plus_noise()
    elif method == 'forecast_open_meteo':
        from ..data_sources.precip_forecast import get_open_meteo_forecast
        df_forecast = get_open_meteo_forecast()
    elif method == 'forecast_knmi':
        from ..data_sources.precip_forecast import get_knmi_forecast
        df_forecast = get_knmi_forecast()
    elif method == 'true_rainfall_knmi':
        df_forecast = build_lag_features(
            df_features.loc[:, [c for c in df_features.columns if 'precip_' in c]],
            lags=[-i for i in range(1, 24)]
        )
        return df_features.merge(df_forecast, left_index=True, right_index=True)
    else:
        return df_features

    # df_shifted_forecast = df_forecast.shift(-24).dropna()
    t_start = df_forecast.index.min()
    t_end = df_forecast.index.max()

    return df_features.loc[t_start: t_end, :].merge(
        df_forecast,
        left_index=True,
        right_index=True,
        how='left',
    )

class ImputationFeatureTransformer(base.BaseEstimator, base.TransformerMixin):
    """Prepare features for iterative imputer.
    """
    def __init__(self, columns=[]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df_lag = engineer_features_for_imputation(
            X.loc[:, self.columns],
            lags_discharge=[1, 2, 3, 4, 5, 8],
            lags_precip=[3],
        )
        return df_lag

class ImputationTransformer(base.BaseEstimator, base.TransformerMixin):
    """Either use pre-fit or re-fit imputation transformer.
    """
    def __init__(self, imputer=IterativeImputer(), refit=False):
        self.imputer = imputer
        self.imputer.verbose = 0
        self.refit = refit

    def fit(self, X, y=None):
        if self.refit:
            self.imputer.fit(X, y)
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)

class ValidationTransformer(base.BaseEstimator, base.TransformerMixin):
    """Remove anomalies and replace with np.nan.
    """
    def __init__(self, validator, refit=False):
        self.validator = validator
        self.refit = refit

    def fit(self, X, y=None):
        if self.refit:
            self.validator.fit(X, y)
        return self

    def transform(self, X):
        return pd.DataFrame(self.validator.predict(X), columns=X.columns, index=X.index)

class PruneFeaturesTransformer(base.BaseEstimator, base.TransformerMixin):
    """Drop specific features.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None,):
        return self

    def transform(self, X):
        if (self.columns_to_drop == ['']) or (self.columns_to_drop == ''):
            return X
        return X.drop(self.columns_to_drop, axis=1).copy()

class SelectOriginalFeaturesTransformer(base.BaseEstimator, base.TransformerMixin):
    """Drop unnecessary features that were used for imputation, but are not needed
    for forecasting.
    """
    def __init__(self, data=None):
        self.data = data

    def fit(self, X, y):
        return self

    def transform(self, X):
        cols_to_use = self.data.columns.difference(X.columns)
        X = X.merge(self.data[cols_to_use], how='left', on='TIME')
        X = X.loc[:, [c for c in X.columns if 'lag' not in c]]
        return X.loc[:, ~X.columns.duplicated()]

class DischargeEngineering(base.BaseEstimator, base.TransformerMixin):
    """ Feature engineering class for Forecasting model Roer
    """
    def __init__(self,
                 discharge_range_max=30,
                 discharge_range_step=1,
                 deficit_long_term_days=60,
                 precip_range_max=16,
                 precip_range_step=1,
                 discharge_stats_power=4,
                 interaction='True',
                 inter_threshold=0.04,
                 keep_forecasts=True,
                 ):

        self.discharge_range_max = discharge_range_max
        self.discharge_range_step = discharge_range_step
        self.deficit_long_term_days = deficit_long_term_days
        self.precip_range_max = precip_range_max
        self.precip_range_step = precip_range_step
        self.discharge_stats_power = discharge_stats_power
        self.interaction = interaction
        self.inter_threshold = inter_threshold
        self.keep_forecasts = keep_forecasts

        self.discharge_range = np.arange(discharge_range_max, step=discharge_range_step)
        self.precip_range = np.arange(precip_range_max, step=precip_range_step)
        self.discharge_stats_range = 2 ** np.arange(2, discharge_stats_power + 1)
        self.discharge_stats_range = [f'{int(x)}D' for x in self.discharge_stats_range]

    def get_feature_names(self):
        return self.feature_names

    def fit(self, X, y=None):
        self.precip_cols = np.sort([x for x in X.columns if 'precip_' in x])
        self.discharge_cols = np.sort([x for x in X.columns if 'discharge_' in x])
        self.forecast_cols = np.sort([x for x in X.columns if 'forecast_' in x])
        self.feature_names = X.columns
        return self

    def transform(self, X, y=None):
        """ Create features
        Parameters
            X: pandas DataFrame
        Returns
            X_out: pandas DataFrame

        """
        if self.keep_forecasts:
            X_out = X.loc[:, self.forecast_cols]
        else:
            X_out = pd.DataFrame()

        if len(self.precip_cols) > 0:
            precip = X.loc[:, self.precip_cols]
            precip_sum = precip.mean(axis=1)
            evap_hour = X.loc[:, 'evap:evap'].divide(24)

            windows = [2, 4, 8, self.deficit_long_term_days]
            for deficit_window in windows:
                deficit = (precip_sum - evap_hour).rolling(f'{deficit_window}D').mean()
                X_out.loc[:, f'deficit_all_mean_{deficit_window}D'] = deficit

            for col in precip.columns:
                for lag in self.precip_range:
                    # lagged precipitation features,
                    # moving average over size of steps, so all information
                    # is remained
                    # TODO: improve backfill?
                    X_out.loc[:, f'{col}_lag_{lag}'] = precip.loc[:, col].shift(lag).bfill()

                    # interaction shortage * precip:
                    if self.interaction == 'True':
                        X_out.loc[:, f'deficit#{col}_lag_{lag}'] = \
                            X_out.loc[:, f'{col}_lag_{lag}'] * (deficit > self.inter_threshold)

        for col in self.discharge_cols:
            for lag in self.discharge_range:
                X_out.loc[:, f'{col}_lag_{lag}'] = X.loc[:, col].shift(lag).bfill()

        # more discharge statistics @ Stah over midterm intervals
        if 'discharge_stah:disc' in X.columns:
            for col in ['stah:disc']:
                for window in self.discharge_stats_range:
                    for rolltype in ['max', 'min', 'mean']:
                        feature = X.loc[:, 'discharge_' + col].rolling(window).agg(rolltype)
                        X_out.loc[:, f'discharge_{col}_{rolltype}_{window}'] = feature

        return X_out


def multicol_output(
    arr: np.ndarray,
    n: int,
    func: Callable[[np.ndarray], np.ndarray],
    fourier: bool = False,
    time_window: str = None,
) -> pd.DataFrame:
    """
    Generic function to compute multiple columns
    func is a function that takes in a numpy array, and outputs a numpy array of the same length
    The numpy array will be a window: e.g. if n=3, func will get a window of 3 points, and output
    3 values. Then, those 3 values will be converted to columns in a dataframe.
    For fourier, an additional column selection is done, so less than 3 columns would be returned
    In that case, the option 'fourier' needs to be set to True

    For nfft, we need not only a numpy array of values, but also a DatetimeIndex. Additionally, we
    have a time_window instead of an integer window. This time_window is for example '100min'.
    If time_window is set, this function will assume that arr has a DatetimeIndex. n is still used
    as the output size.

    Parameters
    ----------
    arr : np.ndarray
        input numpy array to generate multipole columns from.
    n : int
        window size in number of datapoints.
    func : function
        aggregate function to apply to the rolling series.
    fourier : bool, optional
        enable fourier transformation, by default False.
    time_window : str, optional
        size of the time window in pandas timedelta format, by default None.

    Returns
    -------
    pd.DataFrame
        Result after applying a rolling function with size `n` and aggregate function `func`.
    """

    class Helper:
        # https://stackoverflow.com/a/39064656
        def __init__(self, nrow, n):
            if fourier:
                """we are only interested in these coefficients, since the rest is redundant
                See https://en.wikipedia.org/wiki/Discrete_Fourier_transform
                It follows from the definition that when k = 0, the result is simply the sum
                It also follows that the definition for n-k is the same as k, except inverted
                Since we take the absolute value, this inversion is removed, so the result is
                identical. Therefore, we only want the values from k = 1 to k = n//2
                """
                self.useful_coeffs = range(1, n // 2 + 1)
            else:
                self.useful_coeffs = range(0, n)
            ncol = len(self.useful_coeffs)
            self.series = np.full((nrow, ncol), np.nan)
            if time_window is None:
                self.calls = n - 1
            else:
                self.calls = 0
            self.n = n

        def calc_func(self, vector):
            if len(vector) < self.n and time_window is None:
                # We are still at the beginning of the dataframe, nothing to do
                return np.nan
            values = func(vector)
            # If function did not return enough values, pad with zeros
            values = np.pad(values, (0, max(0, self.n - values.size)), "constant")
            values = values[self.useful_coeffs]
            self.series[self.calls, :] = values
            self.calls = self.calls + 1
            return np.nan  # return something to make Rolling apply not error

    helper = Helper(len(arr), n)
    if time_window is None:
        arr.rolling(n, min_periods=0).apply(helper.calc_func, raw=True)
    else:
        # For time-window, we need raw=False because 'func' may need
        # The DatetimeIndex. Even though raw=False is slower.
        arr.rolling(time_window).apply(helper.calc_func, raw=False)
    return pd.DataFrame(helper.series)


class BuildRollingFeatures(base.BaseEstimator, base.TransformerMixin):
    """Applies some rolling function to a pandas dataframe

    This class provides a stateless transformer that applies to each column in a dataframe.
    It works by applying a certain rolling function to each column individually, with a
    window size. The rolling function is given by rolling_type, for example 'mean',
    'median', 'sum', etcetera.

    An important note is that this transformer assumes that the data is sorted by time already!
    So if the input dataframe is not sorted by time (in ascending order), the results will be
    completely wrong.

    A note about the way the output is rolled: in case of 'lag' and 'diff', the output will
    always be lagged, even if lookback is 0. This is because these functions inherently look
    at a previous cell, regardless of what the lookback is. All other functions will start
    by looking at the current cell if lookback is 0. (and will also look at previous cells
    if `window_size` is greater than 1)

    'ewm' looks at `window_size` a bit different: instead of a discrete number of points to
    look at, 'ewm' needs a parameter alpha between 0 and 1 instead.

    Parameters
    ----------
    window_size: array-like, shape = (n_outputs, ), optional (default=None)
        vector of values to shift. Ignored when rolling_type is ewm
        if integer, the window size is fixed, and the timestamps are assumed to be uniform.
        If string of timeoffset (for example '1H'), the input dataframe must have a DatetimeIndex.
        timeoffset is not supported for rolling_type 'lag', 'fourier', 'ewm', 'diff'!
    lookback: number type, optional (default=1)
        the features that are built will be shifted by this value
        If more than 0, this prevents leakage
    rolling_type: string, optional (default="mean")
        The rolling function. Must be one of: 'median', 'skew', 'kurt', 'max', 'std', 'lag',
        'mean', 'diff', 'sum', 'var', 'min', 'numpos', 'ewm', 'fourier', 'cwt', 'trimmean'
    deviation: str, optional (default=None)
        one of ['subtract', 'divide']. If this option is set, the resulting column will either
        have the original column subtracted, or will be divided by the original column. If None,
        just return the resulting column. This option is not allowed when rolling_type is 'cwt'
        or 'fourier', but it is allowed with all other rolling_types.
    alpha: numeric, optional (default=0.5)
        if rolling_type is 'ewm', this is the parameter alpha used for weighing the samples.
        The current sample weighs alpha, the previous sample weighs alpha*(1-alpha), the
        sample before that weighs alpha*(1-alpha)^2, etcetera. Must be in (0, 1]
    width: numeric, optional (default=1)
        if rolling_type is 'cwt', the wavelet transform uses a ricker signal. This parameter
        defines the width of that signal
    nfft_ncol: numeric, optional (default=10)
        if rolling_type is 'nfft', there needs to be a fixed number of columns as output, since
        this is unknown a-priori. This means the number of output-columns will be fixed. If
        nfft has more outputs, and additional outputs are discarded. If nfft has less outputs,
        the rest of the columns are right-padded with 0.
    proportiontocut: numeric, optional (default=0.1)
        if rolling_type is 'trimmean', this is the parameter used to trim values on both tails
        of the distribution. Must be in [0, 0.5). Value 0 results in the mean, close to 0.5
        approaches the median.
    keep_original: boolean, optional (default=True)
        if the original columns should be kept or discarded
        True by default, which means the new columns are added to the old ones
    timecol: str, optional (default=None)
        Optional, the column to set as the index during transform. The index is restored before
        returning. This is only useful when using a timeoffset for window_size, since that needs
        a datetimeindex. So this column can specify a time column. This column will not be
        feature-engineered, and will never be returned in the output!
    add_lookback_to_colname: bool, optional (default=False)
        Whether to add lookback to the newly generated column names.
        if False, column names will be like: DEBIET#mean_2
        if True, column names will be like: DEBIET#mean_2_lookback_0

    Examples
    --------
    >>> from sam.feature_engineering import BuildRollingFeatures
    >>> import pandas as pd
    >>> df = pd.DataFrame({'RAIN': [0.1, 0.2, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ...                    'DEBIET': [1, 2, 3, 4, 5, 5, 4, 3, 2, 4, 2, 3]})
    >>>
    >>> BuildRollingFeatures(rolling_type='lag', window_size = [0,1,4], \\
    ...                      lookback=0, keep_original=False).fit_transform(df)
        RAIN#lag_0  DEBIET#lag_0  ...  RAIN#lag_4  DEBIET#lag_4
    0          0.1             1  ...         NaN           NaN
    1          0.2             2  ...         NaN           NaN
    2          0.0             3  ...         NaN           NaN
    3          0.6             4  ...         NaN           NaN
    4          0.1             5  ...         0.1           1.0
    5          0.0             5  ...         0.2           2.0
    6          0.0             4  ...         0.0           3.0
    7          0.0             3  ...         0.6           4.0
    8          0.0             2  ...         0.1           5.0
    9          0.0             4  ...         0.0           5.0
    10         0.0             2  ...         0.0           4.0
    11         0.0             3  ...         0.0           3.0
    <BLANKLINE>
    [12 rows x 6 columns]
    """

    def __init__(
        self,
        rolling_type: str = "mean",
        lookback: int = 1,
        window_size: Optional[str] = None,
        deviation: Optional[str] = None,
        alpha: float = 0.5,
        width: int = 1,
        nfft_ncol: int = 10,
        proportiontocut: float = 0.1,
        timecol: Optional[str] = None,
        keep_original: bool = True,
        add_lookback_to_colname: bool = False,
    ):
        self.window_size = window_size
        self.lookback = lookback
        self.rolling_type = rolling_type
        self.deviation = deviation
        self.alpha = alpha
        self.width = width
        self.nfft_ncol = nfft_ncol
        self.proportiontocut = proportiontocut
        self.keep_original = keep_original
        self.timecol = timecol
        self.add_lookback_to_colname = add_lookback_to_colname
        logger.debug(
            "Initialized rolling generator. rolling_type={}, lookback={}, "
            "window_size={}, deviation={}, alpha={}, proportiontocut={}, width={}, "
            "keep_original={}, timecol={}".format(
                rolling_type,
                lookback,
                window_size,
                deviation,
                alpha,
                proportiontocut,
                width,
                keep_original,
                timecol,
            )
        )

    def _validate_params(self):
        """apply various checks to the inputs of the __init__ function
        throw value error or type error based on the result
        """

        self._validate_lookback()
        self._validate_width()
        self._validate_alpha()
        self._validate_proportiontocut()

        if not isinstance(self.rolling_type, str):
            raise TypeError("rolling_type must be a string")

        if not isinstance(self.keep_original, bool):
            raise TypeError("keep_original must be a boolean")

        if int(self.nfft_ncol) != self.nfft_ncol:
            raise ValueError("nfft_ncol must be an integer!")

        if self.deviation not in [None, "subtract", "divide"]:
            raise ValueError("Deviation must be one of [None, 'subtract', 'divide']")

        if self.window_size is None and self.rolling_type != "ewm":
            raise ValueError("Window_size must not be None, unless rolling_type is ewm")

        if self.deviation is not None and self.rolling_type in ["fourier", "cwt"]:
            raise ValueError("Deviation cannot be used together with {}".format(self.rolling_type))

    def _validate_lookback(self):
        if not np.isscalar(self.lookback):
            raise TypeError("lookback must be a scalar")
        if self.lookback < 0:
            raise ValueError("lookback cannot be negative!")

    def _validate_width(self):
        if not np.isscalar(self.width):
            raise TypeError("width must be a scalar")
        if self.width <= 0:
            raise ValueError("width must be positive")

    def _validate_alpha(self):
        if not np.isscalar(self.alpha):
            raise TypeError("alpha must be a scalar")
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("alpha must be in (0, 1]")

    def _validate_proportiontocut(self):
        if not np.isscalar(self.proportiontocut):
            raise TypeError("proportiontocut must be a scalar")
        if self.proportiontocut >= 0.5 or self.proportiontocut < 0:
            raise ValueError("proportiontocut must be in [0, 0.5)")

    def _get_rolling_fun(
        self,
        rolling_type: str = "mean",
    ) -> Callable[[Union[pd.Series, np.ndarray], Union[int, None]], Union[pd.Series, np.ndarray]]:
        """Given a function name as a string, creates a function that
        applies that rolling function

        Parameters
        ----------
        rolling_type : string, default="mean"
            the description of the rolling function. must be one of lag, sum, mean,
            median, trimmean, var, std, max, min, skew, kurt, diff, numpos, ewm, fourier, cwt

        Returns
        -------
        rolling_function : function
            function with two inputs: a pandas series and an integer. Will apply
            some rolling function to the series, with window size of the integer.
            Alternatively, in case of fourier/cwt, a function with one input:
            a numpy array. Will output another numpy array.
        """
        if self.rolling_type == "cwt":
            from scipy import signal  # Only needed for this rolling type
        if self.rolling_type == "nfft":
            from nfft import nfft
        if self.rolling_type == "trimmean":
            from scipy.stats import trim_mean

        rolling_functions = {
            "lag": lambda arr, n: arr.shift(n),
            "sum": lambda arr, n: arr.rolling(n).sum(),
            "mean": lambda arr, n: arr.rolling(n).mean(),
            "trimmean": lambda arr, n: arr.rolling(n).apply(
                lambda w: trim_mean(w, self.proportiontocut), raw=True
            ),
            "median": lambda arr, n: arr.rolling(n).median(),
            "var": lambda arr, n: arr.rolling(n).var(),
            "std": lambda arr, n: arr.rolling(n).std(),
            "max": lambda arr, n: arr.rolling(n).max(),
            "min": lambda arr, n: arr.rolling(n).min(),
            "skew": lambda arr, n: arr.rolling(n).skew(),
            "kurt": lambda arr, n: arr.rolling(n).kurt(),
            "diff": lambda arr, n: arr.diff(n),
            "numpos": lambda arr, n: arr.gt(0).rolling(n).sum(),
            "ewm": lambda arr, n: arr.ewm(alpha=self.alpha).mean(),
            # These two have different signature because they are called by multicol_output
            "fourier": lambda vector: np.absolute(np.fft.fft(vector)),
            "cwt": lambda vector: signal.cwt(vector, signal.ricker, [self.width])[0],
        }

        if rolling_type not in rolling_functions:
            raise ValueError(
                "The rolling_type is %s, which is not an available function" % rolling_type
            )

        return rolling_functions[rolling_type]

    def _apply_deviation(
        self, arr: np.ndarray, original: np.ndarray, deviation: str
    ) -> np.ndarray:
        """Helper function to apply deviation during the transform"""
        if deviation is None:
            return arr
        if deviation == "subtract":
            return arr - original
        if deviation == "divide":
            # Pandas will insert inf when dividing by 0
            return arr / original

    def _generate_and_add_new_features(
        self, X: pd.DataFrame, result: pd.DataFrame
    ) -> pd.DataFrame:
        """Applies rolling functions to pandas dataframe `X` and concatenates result to `result`.

        Parameters:
        ----------
        X: pandas dataframe
           the pandas dataframe that you want to apply rolling functions on
        result: pandas dataframe
           the pandas dataframe that you want to add the new features to

        Returns
        -------
        pandas dataframe, shape = `(n_rows, n_features * (n_outputs + 1))`
            the pandas dataframe, appended with the new feature columns
        """

        if self.rolling_type in ["fourier", "cwt", "nfft"]:
            for window_size, suffix in zip(self.window_size_, self.suffix_):
                # If rolling type is nfft, the time_window needs to be set
                if self.rolling_type == "nfft":
                    window_size, time_window = self.nfft_ncol, window_size
                else:
                    time_window = None

                for column in X.columns:
                    new_features = multicol_output(
                        X[column],
                        window_size,
                        self.rolling_fun_,
                        self.rolling_type == "fourier",
                        time_window=time_window,
                    ).shift(self.lookback)
                    # Fourier has less columns
                    if self.rolling_type == "fourier":
                        useful_coeffs = range(1, window_size // 2 + 1)
                    else:
                        useful_coeffs = range(0, window_size)
                    col_prefix = "#".join([str(column), suffix])
                    new_features.columns = ["_".join([col_prefix, str(j)]) for j in useful_coeffs]
                    new_features = new_features.set_index(X.index)
                    result = pd.concat([result, new_features], axis=1)
        else:
            for window_size, suffix in zip(self.window_size_, self.suffix_):
                new_features = X.apply(
                    lambda arr: self._apply_deviation(
                        self.rolling_fun_(arr, window_size).shift(self.lookback),
                        arr,
                        self.deviation,
                    ),
                    raw=False,
                )
                new_features.columns = [
                    "#".join([str(col), suffix]) for col in new_features.columns
                ]
                result = pd.concat([result, new_features], axis=1)

        return result

    def fit(self, X: Any = None, y: Any = None):
        """Calculates window_size and feature function

        Parameters
        ----------
        X: optional, is ignored
        y: optional, is ignored
        """

        self._validate_params()

        if self.rolling_type == "ewm":
            # ewm needs no integer window_size
            self.window_size_ = "ewm"
            self.suffix_ = ["ewm_" + str(self.alpha)]
        else:
            self.window_size_ = self.window_size
            # Singleton window_size is also allowed
            if np.isscalar(self.window_size_):
                self.window_size_ = [self.window_size_]
            self.suffix_ = [
                self.rolling_type + "_" + str(window_size) for window_size in self.window_size_
            ]
            if self.add_lookback_to_colname:
                self.suffix_ = [s + "_lookback_" + str(self.lookback) for s in self.suffix_]
        self.rolling_fun_ = self._get_rolling_fun(self.rolling_type)
        logger.debug(
            "Done fitting transformer. window size: {}, suffix: {}".format(
                self.window_size_, self.suffix_
            )
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms pandas dataframe `X` to apply rolling function

        Parameters
        ----------
        X: pandas dataframe, shape = `(n_rows, n_features)`
           the pandas dataframe that you want to apply rolling functions on

        Returns
        -------
        result: pandas dataframe, shape = `(n_rows, n_features * (n_outputs + 1))`
            the pandas dataframe, appended with the new columns
        """

        if self.keep_original:
            result = X.copy()
        else:
            result = pd.DataFrame(index=X.index)

        if self.timecol is not None:
            # Set DatetimeIndex on the intermediate result
            index_backup = X.index.copy()
            new_index = pd.DatetimeIndex(X[self.timecol].values)
            X = X.set_index(new_index).drop(self.timecol, axis=1)
            if self.keep_original:
                result = result.set_index(new_index).drop(self.timecol, axis=1)
            else:
                result = pd.DataFrame(index=new_index)

        result = self._generate_and_add_new_features(X, result)

        self._feature_names = list(result.columns.values)
        if self.timecol is not None:
            result = result.set_index(index_backup)

        return result

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns feature names for the outcome of the last transform call.
        """
        return self._feature_names


@dataclass
class CyclicalMaxes:
    """Class for keeping track of maximum integer values of specific cyclicals"""

    day: int = 31
    dayofweek: int = 7
    weekday: int = 7
    dayofyear: int = 366
    hour: int = 24
    microsecond: int = 1000000
    minute: int = 60
    month: int = 12
    quarter: int = 4
    second: int = 60
    week: int = 53
    secondofday: int = 86400

    @classmethod
    def get_maxes_from_strings(cls, cyclicals: Sequence[str]) -> List[int]:
        """
        This method retrieves cyclical_maxes for pandas datetime features
        The CyclicalMaxes class contains maxes for those features that are actually cyclical.
        For example, 'year' is not cyclical so is not included here.
        Note that the maxes are chosen such that these values are equivalent to 0.
        e.g.: a minute of 60 is equivalent to a minute of 0
        For month, dayofyear and week, these are approximations, but close enough.

        Parameters
        ----------
        cyclicals : list
            List of cyclical strings that match attributes in self class

        Returns
        -------
        list
            List of integer representations of the cyclicals
        """
        for c in cyclicals:
            if c not in cls.__annotations__:
                raise ValueError(
                    str(c) + " is not a known cyclical, please " "provide cyclical_maxes yourself."
                )
        return [getattr(cls, c.lower()) for c in cyclicals]


def decompose_datetime(
    df: pd.DataFrame,
    column: Optional[str] = "TIME",
    components: Optional[Sequence[str]] = None,
    cyclicals: Optional[Sequence[str]] = None,
    onehots: Optional[Sequence[str]] = None,
    remove_categorical: bool = True,
    keep_original: bool = True,
    cyclical_maxes: Optional[Sequence[int]] = None,
    cyclical_mins: Optional[Union[Sequence[int], int]] = (0,),
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Decomposes a time column to one or more components suitable as features.

    The input is a dataframe with a pandas timestamp column. New columns will be added to this
    dataframe. For example, if column is 'TIME', and components is ['hour', 'minute'], two
    columns: 'TIME_hour' and 'TIME_minute' will be added.

    Optionally, cyclical features can be added instead. For example, if cyclicals is ['hour'],
    then the 'TIME_hour' column will not be added, but two columns 'TIME_hour_sin' and
    'TIME_hour_cos' will be added instead. If you want both the categorical and cyclical features,
    set 'remove_categorical' to False.

    Parameters
    ----------
    df: dataframe
        The dataframe with source column
    column: str (default='TIME')
        Name of the source column to extract components from. Note: time column should have a
        datetime format. if None, it is assumed that the TIME column will be the index.
    components: list
        List of components to extract from datatime column. All default pandas dt components are
        supported, and some custom functions: `['secondofday', 'week']`.
        Note: `week` was added here since it is deprecated in pandas in favor of
        `isocalendar().week`
    cyclicals: list
        List of strings of newly created .dt time variables (like hour, month) you want to convert
        to cyclicals using sine and cosine transformations. Cyclicals are variables that do not
        increase linearly, but wrap around, such as days of the week and hours of the day.
        Format is identical to `components` input.
    onehots: list
        List of strings of newly created .dt time variables (like hour, month) you want to convert
        to one-hot-encoded variables. This is suitable when you think that variables do not
        vary smoothly with time (e.g. Sunday and Monday are quite different).
        This list must be mutually exclusive from cyclicals, i.e. non-overlapping.
    remove_categorical: bool, optional (default=True)
        whether to keep the original cyclical features (i.e. day)
        after conversion (i.e. day_sin, day_cos)
    keep_original: bool, optional (default=True)
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original ones
    cyclical_maxes: sequence, optional (default=None)
        Passed through to recode_cyclical_features. See :ref:`recode_cyclical_features` for more
        information.
    cyclical_mins: sequence or int, optional (default=0)
        Passed through to recode_cyclical_features. See :ref:`recode_cyclical_features` for more
        information.
    timezone: str, optional (default=None)
        if tz is not None, convert the time to the specified timezone, before creating features.
        timezone can be any string that is recognized by pytz, for example `Europe/Amsterdam`.
        We assume that the TIME column is always in UTC,
        even if the datetime object has no tz info.
    Returns
    -------
    dataframe
        The original dataframe with extra columns containing time components

    Examples
    --------
    >>> from sam.feature_engineering import decompose_datetime
    >>> import pandas as pd
    >>> df = pd.DataFrame({'TIME': pd.date_range("2018-12-27", periods = 4),
    ...                    'OTHER_VALUE': [1, 2, 3, 2]})
    >>> decompose_datetime(df, components= ["year", "dayofweek"])
            TIME  OTHER_VALUE  TIME_year  TIME_dayofweek
    0 2018-12-27            1       2018               3
    1 2018-12-28            2       2018               4
    2 2018-12-29            3       2018               5
    3 2018-12-30            2       2018               6
    """
    components = [] if components is None else components
    cyclicals = [] if cyclicals is None else cyclicals
    onehots = [] if onehots is None else onehots
    cyclical_maxes = [] if cyclical_maxes is None else cyclical_maxes

    if np.any([c in cyclicals for c in onehots]):
        raise ValueError("cyclicals and onehots are not mutually exclusive")

    if keep_original:
        result = df.copy()
    else:
        result = pd.DataFrame(index=df.index)

    if column is None:
        timecol = df.index.to_series().copy()
        column = "" if timecol.name is None else timecol.name
    else:
        timecol = df[column].copy()

    logging.debug(
        f"Decomposing datetime, number of dates: {len(timecol)}. " f"Components: {components}"
    )

    # Fix timezone
    if timezone is not None:
        if timecol.dt.tz is not None:
            if timecol.dt.tz != pytz.utc:
                raise ValueError(
                    "Data should either be in UTC timezone or it should have no"
                    " timezone information (assumed to be in UTC)"
                )
        else:
            timecol = timecol.dt.tz_localize("UTC")
        timecol = timecol.dt.tz_convert(timezone)

    result = _create_time_cols(result, components, timecol, column)

    # convert cyclicals
    if not isinstance(cyclicals, Sequence):
        raise TypeError("cyclicals must be a sequence type")
    if cyclicals:
        result = recode_cyclical_features(
            result,
            cyclicals,
            prefix=column,
            remove_categorical=remove_categorical,
            cyclical_maxes=cyclical_maxes,
            cyclical_mins=cyclical_mins,
            keep_original=True,
        )
    if onehots:
        result = recode_onehot_features(
            result,
            onehots,
            prefix=column,
            remove_categorical=remove_categorical,
            onehot_maxes=cyclical_maxes,
            onehot_mins=cyclical_mins,
            keep_original=True,
        )

    return result


def recode_cyclical_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    prefix: str = "",
    remove_categorical: bool = True,
    keep_original: bool = True,
    cyclical_maxes: Optional[Sequence[int]] = None,
    cyclical_mins: Optional[Union[Sequence[int], int]] = (0,),
) -> pd.DataFrame:
    """
    Convert cyclical features (like day of week, hour of day) to continuous variables, so that
    Sunday and Monday are close together numerically.

    IMPORTANT NOTE: This function requires a global maximum and minimum for the data. For example,
    for minutes, the global maximum and minimum are 0 and 60 respectively, even if your data never
    reaches these global minimums/maximums explicitly. This function assumes that the minimum and
    maximum should be encoded as the same value: minute 0 and minute 60 mean the same thing.

    If you only use cyclical pandas timefeatures, nothing needs to be done. For these features,
    the minimum/maximum will be chosen automatically. These are: ['day', 'dayofweek', 'weekday',
    'dayofyear', 'hour', 'microsecond', 'minute', 'month', 'quarter', 'second', 'week']

    For any other scenario, global minimums/maximums will need to be passed in the parameters
    `cyclical_maxes` and `cyclical_mins`. Minimums are set to 0 by default, meaning that
    only the maxes need to be chosen as the value that is `equivalent` to 0.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe in which the columns to convert should be present.
    cols: list of strings
        The suffixes column names to convert to continuous numerical values.
        These suffixes will be added to the `column` argument to get the actual column names, with
        a '_' in between.
    column: string, optional (default='')
        name of original time column in df, e.g. TIME.
        By default, assume the columns in cols literally refer to column names in the data
    remove_categorical: bool, optional (default=True)
        whether to keep the original cyclical features (i.e. day)
        after conversion (i.e. day_sin, day_cos)
    keep_original: bool, optional (default=True)
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original
        ones. If `remove_categorical` is False, the categoricals will be kept, regardless of
        this argument.
    cyclical_maxes: array-like, optional (default=None)
        The maximums that your data can reach. Keep in mind that the maximum value and the
        minimum value will be encoded as the same value. By default, None means that only
        standard pandas timefeatures will be encoded.
    cyclical_mins: array-like or scalar, optional (default=[0])
        The minimums that your data can reach. Keep in mind that the maximum value and the
        minimum value will be encoded as the same value. By default, 0 is used, which is
        applicable for all pandas timefeatures.

    Returns
    -------
    dataframe
        The input dataframe with cols removed, and replaced by the converted features (two for
        each feature).
    """

    new_df, prefix, cyclical_maxes, cyclical_mins = _validate_and_prepare_components(
        df=df,
        cols=cols,
        column=prefix,
        remove_categorical=remove_categorical,
        keep_original=keep_original,
        component_maxes=cyclical_maxes,
        component_mins=cyclical_mins,
    )

    logging.debug(f"Sine/cosine converting cyclicals columns: {cols}")

    for cyclical_min, cyclical_max, col in zip(cyclical_mins, cyclical_maxes, cols):
        if cyclical_min >= cyclical_max:
            raise ValueError(
                "Cyclical min {} is higher than cyclical max {} for column {}".format(
                    cyclical_min, cyclical_max, col
                )
            )

        # prepend column name (like TIME) to match df column names
        col = prefix + col

        if col not in df.columns:
            raise ValueError(f"{col} is not in input dataframe")

        # rescale feature so it runs from 0 to 2*pi:
        # Features that exceed the maximum are rolled over by the sine/cosine:
        # e.g. if min=0 and max=7, 9 will be treated the same as 2
        norm_feature: pd.Series = (df[col] - cyclical_min) / (cyclical_max - cyclical_min)
        norm_feature = 2 * np.pi * norm_feature
        # convert cyclical to 2 variables that are offset:
        new_df[col + "_sin"] = np.sin(norm_feature)
        new_df[col + "_cos"] = np.cos(norm_feature)

        # drop the original. if keep_original is False, this is unneeded: it was already removed
        if remove_categorical and keep_original:
            new_df = new_df.drop(col, axis=1)

    return new_df


def recode_onehot_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    prefix: str = "",
    remove_categorical: bool = True,
    keep_original: bool = True,
    onehot_maxes: Optional[Sequence[int]] = None,
    onehot_mins: Optional[Union[Sequence[int], int]] = (0,),
) -> pd.DataFrame:
    """
    Convert time features (like day of week, hour of day) to onehot variables (1 or 0 for each
    unique value).

    IMPORTANT NOTE: This function requires a global maximum and minimum for the data. For example,
    for minutes, the global maximum and minimum are 0 and 60 respectively, even if your data never
    reaches these global minimums/maximums explicitly. Make sure these variables will all be added
    in your onehot columns, otherwise your columns in train and test set could be unmatching.

    If you only use cyclical pandas timefeatures, nothing needs to be done. For these features,
    the minimum/maximum will be chosen automatically. These are: ['day', 'dayofweek', 'weekday',
    'dayofyear', 'hour', 'microsecond', 'minute', 'month', 'quarter', 'second', 'week']

    For any other scenario, global minimums/maximums will need to be passed in the parameters
    `cyclical_maxes` and `cyclical_mins`. Minimums are set to 0 by default, meaning that
    only the maxes need to be chosen as the value that is `equivalent` to 0.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe in which the columns to convert should be present.
    cols: list of strings
        The suffixes column names to convert to onehot variables.
        These suffixes will be added to the `column` argument to get the actual column names, with
        a '_' in between.
    prefix: string, optional (default='')
        name of original time column in df, e.g. 'TIME'.
        By default, assume the columns in cols literally refer to column names in the data
    remove_categorical: bool, optional (default=True)
        whether to keep the original time features (i.e. day)
    keep_original: bool, optional (default=True)
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original
        ones. If `remove_categorical` is False, the categoricals will be kept, regardles of
        this argument.
    onehot_maxes: array-like, optional (default=None)
        The maximums that your data can reach. By default, None means that only
        standard pandas timefeatures will be encoded.
    onehot_mins: array-like or scalar, optional (default=[0])
        The minimums that your data can reach. By default, 0 is used, which is
        applicable for all pandas timefeatures.

    Returns
    -------
    pandas dataframe
        The input dataframe with cols removed, and replaced by the converted features.
    """

    new_df, prefix, onehot_maxes, onehot_mins = _validate_and_prepare_components(
        df=df,
        cols=cols,
        column=prefix,
        remove_categorical=remove_categorical,
        keep_original=keep_original,
        component_maxes=onehot_maxes,
        component_mins=onehot_mins,
    )

    logging.debug(f"onehot converting time columns: {cols}")

    for onehot_min, onehot_max, col in zip(onehot_mins, onehot_maxes, cols):
        col = prefix + col

        if col not in df.columns:
            raise ValueError(f"{col} is not in input dataframe")

        # get the onehot encoded dummies
        dummies: pd.DataFrame = pd.get_dummies(df[col], prefix=col).astype(int)

        # fill in the weekdays not in the dataset
        for i in range(onehot_min, onehot_max):
            if not "%s_%d" % (col, i) in dummies.columns:
                dummies["%s_%d" % (col, i)] = 0
        dummies_sorted = dummies[np.sort(dummies.columns)]
        new_df = new_df.join(dummies_sorted)

        # drop the original. if keep_original is False, this is unneeded: it was already removed
        if remove_categorical and keep_original:
            new_df = new_df.drop(col, axis=1)

    return new_df


def _create_time_cols(
    df: pd.DataFrame, components: Sequence[str], timecol: pd.Series, prefix: str = ""
) -> pd.DataFrame:
    """Helper function to create all the neccessary time columns

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe in SAM format
    components : Sequence[str]
        Time components that need to be added to the data
    timecol: Series
        A pandas series containing the datetimes, used for making the time columns
    prefix : str
        Prefix of the newly created columns, usually the same as the original time column

    Returns
    -------
    pd.DataFrame
        The dataframe, which includes the time components

    Raises
    ------
    NotImplementedError
        In case the time components are not recognized by pandas or by SAM
    """

    pandas_functions = [f for f in dir(timecol.dt) if not f.startswith("_")]

    custom_functions = ["secondofday", "week"]
    for component in components:
        if component in custom_functions:
            if component == "week":
                df[prefix + "_" + component] = timecol.dt.isocalendar().week
            elif component == "secondofday":
                sec_in_min = 60
                sec_in_hour: int = sec_in_min * 60
                df[prefix + "_" + component] = (
                    timecol.dt.hour * sec_in_hour
                    + timecol.dt.minute * sec_in_min
                    + timecol.dt.second
                )
        elif component in pandas_functions:
            df[prefix + "_" + component] = getattr(timecol.dt, component)
        else:
            raise NotImplementedError(f"Component {component} not implemented")
    return df


def _validate_and_prepare_components(
    df: pd.DataFrame,
    cols: Sequence[str],
    column: str,
    remove_categorical: bool,
    keep_original: bool,
    component_maxes: Optional[Sequence[int]],
    component_mins: Optional[Union[Sequence[int], int]],
) -> Tuple[pd.DataFrame, str, Sequence[int], Sequence[int]]:
    """
    Validates and prepares the dataframe, component (onehot or cyclical) parameters and min/max
    component bounds.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe in which the columns to convert should be present.
    cols: list of strings
        The suffixes column names to convert to onehot variables.
        These suffixes will be added to the `column` argument to get the actual column names, with
        a '_' in between.
    column: string
        name of original time column in df (e.g. TIME)
        By default, assume the columns in cols literally refer to column names in the data
    remove_categorical: bool
        whether to keep the original time features (i.e. day)
    keep_original: bool
        whether to keep the original columns from the dataframe. If this is False, then the
        returned dataframe will only contain newly generated columns, and none of the original
        ones. If `remove_categorical` is False, the categoricals will be kept, regardles of
        this argument.
    component_maxes: array-like
        The maximums that your data can reach. By default, None means that only
        standard pandas timefeatures will be encoded.
    component_mins: array-like or scalar
        The minimums that your data can reach. By default, 0 is used, which is
        applicable for all pandas timefeatures.

    Returns
    -------
    new_df: pandas dataframe
        Dataframe with/without the categoricals and the original columns depending on setttings.
    column: string
        updated name of original time column in df, e.g. 'TIME_'.
    component_maxes: array-like
        The maximums that your data can reach.
    component_mins: array-like
        The minimums that your data can reach.
    """

    column = column + "_" if column != "" else column
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be pandas dataframe")
    if not isinstance(cols, Sequence):
        raise TypeError("cols should be a sequence of columns to convert")
    if not isinstance(remove_categorical, bool):
        raise TypeError("remove_categorical should be a boolean")

    if component_maxes is None or not component_maxes:
        component_maxes = CyclicalMaxes.get_maxes_from_strings(cols)

    if np.isscalar(component_mins):
        component_mins = [cast(int, component_mins)] * len(component_maxes)
    elif isinstance(component_mins, (Sequence, np.ndarray)):
        if len(component_mins) == 1:
            component_mins = list(component_mins)
            component_mins *= len(component_maxes)
    else:
        raise TypeError("`component_maxes` needs to be a scalar or array-like")

    if keep_original:
        new_df = df.copy()
    elif not remove_categorical:
        # We don't want to keep the original columns, but we want to keep the categoricals
        new_df = df.copy()[[column + col for col in cols]]
    else:
        new_df = pd.DataFrame(index=df.index)

    return new_df, column, component_maxes, component_mins
