import pandas as pd
from datetime import datetime

from .anomaly_detection import ValidationModel


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
