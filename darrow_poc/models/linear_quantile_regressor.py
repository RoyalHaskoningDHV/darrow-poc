import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# Keep package independent of statsmodels
try:
    from statsmodels.regression.quantile_regression import QuantReg
except ImportError:
    pass


class LinearQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    scikit-learn style wrapper for QuantReg
    Fits a linear quantile regression model, base idea from
    https://github.com/Marco-Santoni/skquantreg/blob/master/skquantreg/quantreg.py
    This class requires statsmodels

    Parameters
    ----------
    quantiles: list or float, default=[0.05, 0.95]
        Quantiles to fit, with `` 0 < q < 1 `` for each q in quantiles.
    tol: float, default=1e-3
        The tolerance for the optimization. The optimization stops
        when duality gap is smaller than the tolerance
    max_iter: int, default=1000
        The maximum number of iterations
    fit_intercept: bool, default=True
        Whether to calculate the intercept for this model. If set to false,
        no intercept will be used in calculations (e.g. data is expected to be
        already centered).

    Attributes
    ----------
    model_: statsmodel model
        The underlying statsmodel class
    model_result_: statsmodel results
        The underlying statsmodel results

    Examples
    --------
    >>> # Prepare data
    >>> data = read_knmi('2018-02-01', '2019-10-01', freq='hourly',
    >>>                 variables=['FH', 'FF', 'FX', 'T']).set_index('TIME')
    >>> y = data['T']
    >>> X = data.drop('T', axis=1)
    >>> # Fit model
    >>> model = LinearQuantileRegression()
    >>> model.fit(X, y)
    """

    def __init__(
        self,
        quantiles: list = [0.05, 0.95],
        tol: float = 1e-3,
        max_iter: int = 1000,
        fit_intercept: bool = True,
    ):
        self.quantiles = quantiles
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def _fit_single_model(self, X, y, q):
        # Statsmodels requires to add constant columns manually
        # otherwise the models will not fit an intercept
        X = pd.DataFrame(X).copy()
        if self.fit_intercept:
            X = X.assign(const=1)  # f(x) = a + bX = a*const + b*X
        model_ = QuantReg(y, X)
        model_result_ = model_.fit(q, p_tol=self.tol, max_iter=self.max_iter)
        coef = model_result_.params
        pvalues = model_result_.pvalues
        return coef, pvalues

    def fit(self, X: np.array, y: np.array):
        """
        Fit a Linear Quantile Regression using statsmodels
        """
        if type(self.quantiles) is float:
            self.q_ = [self.quantiles]
        elif type(self.quantiles) is list:
            self.q_ = self.quantiles
        else:
            raise TypeError(f"Invalid type, quantile {self.quantiles} " f"should be a float or list of floats")
        self.prediction_cols = [f"predict_q_{q}" for q in self.quantiles]
        models_ = [self._fit_single_model(X, y, q) for q in self.quantiles]
        self.coef_ = [m[0] for m in models_]
        self.pvalue_ = [m[1] for m in models_]
        return self

    def predict(self, X: np.array):
        """
        Predict / estimate quantiles
        """
        preds = [pd.DataFrame(X).assign(const=1).multiply(c).sum(axis=1) for c in self.coef_]
        preds_df = pd.concat(preds, axis=1)
        preds_df.columns = self.prediction_cols
        return preds_df

    def score(self, X: np.array, y: np.array):
        """
        Default score function. Returns the tilted loss
        """
        y_pred = self.predict(X)
        scores = [tilted_loss(y_true=y, y_pred=y_pred[f"predict_q_{q}"], quantile=q) for q in self.quantiles]
        score = np.mean(scores)
        return score


def tilted_loss(y_true: np.array, y_pred: np.array, quantile: float = 0.5):
    """
    Calculate tilted, or quantile loss with numpy. Given a quantile q, and an error e,
    then tilted loss is defined as `(1-q) * |e|` if `e < 0`, and `q * |e|` if `e > 0`.

    This function is the same as the mean absolute error if q=0.5, which approximates the median.
    For a given value of q, the function that minimizes the tilted loss will be the q'th quantile
    function.

    Parameters
    ----------
    y_true: array-like of shape = (n_samples)
        True labels.
    y_pred: array-like of shape = (n_samples)
        Predictions. Must be same shape as `y_true`
    quantile: float, optional (default=0.5)
        The quantile to use when computing tilted loss.
        Must be between 0 and 1 for tilted loss to be positive.

    Returns
    -------
    float:
        The quantile loss

    Examples
    --------
    >>> import numpy as np
    >>> from sam.metrics import tilted_loss
    >>> actual = np.array([1, 2, 3, 4])
    >>> pred = np.array([0.9, 2.1, 2.9, 3.1])
    >>> tilted_loss(actual, pred, quantile=0.5)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    e = y_true - y_pred
    return np.mean(np.maximum(quantile * e, (quantile - 1) * e), axis=-1)
