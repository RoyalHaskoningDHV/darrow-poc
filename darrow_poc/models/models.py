import pandas as pd
import numpy as np
import dill as pkl
from pathlib import Path
from datetime import datetime
import logging

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import KFold, RandomizedSearchCV

from .feature_engineering import (
    create_target,
    engineer_features_for_imputation,
    ValidationTransformer,
    ImputationFeatureTransformer,
    ImputationTransformer,
    SelectOriginalFeaturesTransformer,
    PruneFeaturesTransformer,
    DischargeEngineering,
    add_precipitation_forecast_features,
)
from .anomaly_detection import ValidationModel
from .quantreg import LinearQuantileRegressor
from ..metrics import calculate_metrics, get_scorer


def train_validator(
    df: pd.DataFrame,
    model_type: str = 'lasso',
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

    if testing is False:
        with open(f'output/validation/model_{timestamp}.pkl', 'wb') as file:
            pkl.dump(validator, file)

    return validator, timestamp


def train_imputer(
    train,
    lags_discharge=[1, 2, 3, 4, 5, 8],
    lags_precip=[3],
    testing=False,
):
    """train imputation model on `train` data, with specified lags.
    Write result to file.

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    lags_discharge : list (default=[1, 2, 3, 4, 5, 8])
        Lagged features to create for discharge sensors
    lags_precip : list (default=[3])
        Lagged features to create for worm and upper roer precip.
        sensors.
    testing : bool (default=False)
        Only set to True when running tests, ommits writing data to file,
        returns model instead.

    Returns
    -------
    imputer : sklearn.Pipeline
        Pipeline of trained imputation model.
    timestamp : str
        Returning the timestamp makes it easy to immediately continue with this
        model, also wenn loading it from file.
    """
    df_lag_train = engineer_features_for_imputation(
        train,
        lags_discharge=lags_discharge,
        lags_precip=lags_precip,
    )

    # Pre-train imputer on channel selection of interest
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        sample_posterior=True,
        max_iter=10,
        initial_strategy='median',
        verbose=0,
        min_value=0,
        n_nearest_features=15,
    )
    imputer.fit(df_lag_train)

    timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")

    if testing is False:
        with open(f'output/imputation/model_{timestamp}.pkl', 'wb') as file:
            pkl.dump(imputer, file)

    return imputer, timestamp


def create_model_save_name(model_name):
    """From a model name (which is a tuple of missing channels),
    create a sensible str to append to the save filename.

    Parameters
    ----------
    model_name : Tuple
        Tuple of strings of missing channels

    Returns
    -------
    model_save_name : str
        Readible and sensibly short string to add to model filename.
    """
    model_save_name = str(model_name).replace('(', '').replace(")", ''
        ).replace("'", '').replace(', ', '_'
        ).replace('discharge_', '').replace(',', '')
    if 'precip' in model_save_name:
        model_save_name = 'precip'

    return model_save_name


def prepare_data_for_forecasting_model(
    data,
    horizons=np.arange(1, 25),
):
    """Prepare data for forecasting model - i.e. divide into
    feature matrix and target matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Input data to conver to X and y
    horizons : np.arange (default=np.arange(1, 25))
        Horizons for which to make predictions (in samples from now,
        which should be in hours)

    Returns
    -------
    y : pd.DataFrame
        Target variable matrix (each column is one horizon in hours leading)
    X : pd.DataFrame
        Feature matrix
    """
    # prepare data
    horizon_max = max(horizons)
    y_train = create_target(data, horizons)

    # drop observation without horizon
    y, X = y_train.iloc[:-horizon_max], data.iloc[:-horizon_max]

    return y, X


def train_forecasting_models(
    train,
    validator,
    imputer,
    timestamp='2021-12-22T09.52.58',
    missing_features=[['']],
    testing=False,
    model_type: str = 'ridge',
    params={},
):
    """Train forecasting models.

    Parameters
    ----------
    train : pd.DataFrame
        Training data
    validator : darrow_poc.models.anomaly_detection.ValidationModel
        Anomaly detection class (trained already)
    imputer : sklearn.IterativeImputer
        Already trained iterative imputer instance
    timestamp : str
        Useful to pass the same value as the one in the filename of the imputer you use.
        Used for saving the output.
    missing_features : list
        List elements are lists of strings, denoting the channels to ommit from model training.
        That means the reduced models will have fewer channels.
    testing : bool (default=False)
        Only set to true when running tests, does not save models when True.
    model_type: str (default='ridge')
        Possible values: 'ridge', 'lasso'
    params : dict (default={})
        Paramters to pass to the forecasting model cross validation

    Returns
    -------
    full_models : dict
        Dictionary keys are the tuples of missing channel names identifying a given model,
        values are the full model pipelines, including the imputer and trained forecaster.
    """

    # Add model where all precipitation channels are missing
    missing_features.append([c for c in train.columns if 'precip' in c])
    if testing is True:
        missing_features = missing_features[0]

    # Replace anomalies with np.nan
    pipe_validate = ValidationTransformer(validator=validator)
    train_validated = pipe_validate.transform(train)

    # Ensure missing values are imputed by trained imputer
    pipe_impute = Pipeline([
        ('ift', ImputationFeatureTransformer(
            columns=[c for c in train.columns if ('discharge' in c) or ('precip' in c)]
        )),
        ('it', ImputationTransformer(imputer=imputer, refit=False)),
        ('soft', SelectOriginalFeaturesTransformer(train)),
    ])
    train_imputed = pipe_impute.transform(train_validated)

    # Get target and feature matrices
    y_train, X_train = prepare_data_for_forecasting_model(train_imputed)

    models = dict()
    for i, columns_to_drop in enumerate(missing_features):

        logging.info(i, columns_to_drop)

        if testing is True:
            if i > 0:
                break

        # Grid search
        if model_type == 'lasso':
            pipe_forecast = Pipeline([
                ('prune', PruneFeaturesTransformer(columns_to_drop)),
                ('features', DischargeEngineering()),
                ('scaler', StandardScaler()),
                ('lm', Lasso()),
            ])
            param_grid = dict(
                lm__tol=[0.01],
                lm__max_iter=[5000],
                lm__alpha=loguniform(0.1, 1),
            )
        elif model_type == 'ridge':
            pipe_forecast = Pipeline([
                ('prune', PruneFeaturesTransformer(columns_to_drop)),
                ('features', DischargeEngineering()),
                ('scaler', StandardScaler()),
                ('select', SelectFromModel(Ridge())),
                ('lm', Ridge()),
            ])
            param_grid = dict(
                select__max_features=[200],
                select__estimator__alpha=[1e-6],
                lm__alpha=loguniform(1e-8, 1e-2),
            )
        else:
            raise NotImplementedError(
                "Only 'lasso' and 'ridge' are currently implemented as model_type"
            )

        # Cross validation
        cv = KFold(2, shuffle=False)
        scoring = get_scorer()
        model = RandomizedSearchCV(pipe_forecast,
                                   param_grid,
                                   cv=cv,
                                   n_iter=2,
                                   scoring=scoring,
                                   refit='wmae',
                                   verbose=2,
                                   n_jobs=1,
                                   random_state=13)
        model.fit(X_train, np.array(y_train))

        models[tuple(sorted(set(columns_to_drop)))] = model

    # Add imputer and save full models (validator + imputer + forecaster pipelines)
    full_models = dict()
    for model_name, model in models.items():

        full_models[model_name] = Pipeline([
            ('validator', pipe_validate),
            ('imputer', pipe_impute),
            ('estimator', model.best_estimator_),
        ])

    if testing is False:
        with open(f'output/models/full_models_{timestamp}.pkl', 'wb') as file:
            pkl.dump(full_models, file)

    return full_models


def train_quantile_models(
    train=None,
    full_models=None,
    data_version='20211215',
    timestamp='2021-12-22T09.52.58',
    testing=False,
):
    """Train quantile regression models for 0.05, 0.25, 0.75 and 0.95 quantiles
    for each of the existing mean prediction models.

    Parameters
    ----------
    train : pd.DataFrame or None (default=None)
        Training data. When None, load training data using `data_version`. Note,
        we assume that training data has already been through the imputer.
    full_models : dict or None (default=None)
        When none, loads full_models from file using `timestamp`.
    data_version : str (default='20211215')
        Parameter set when saving the train and test data. Select desired version
        (see train.ipynb)
    timestamp : str (default='2021-12-22T09.52.58')
        Parameter set when saving the mean prediction models, so fill in the value
        of the version you desire to train quantile models for (see train.ipynb)
    testing : bool (default=False)
        When not testing, results are pickled / cached. When testing they are returned
        instead.

    Returns
    -------
    quantile_models : dict
        Keys are the model names, containing dictionaries with quantile names as keys,
        which then contain the corresponding trained model obeject.
    """

    # Get forecasting models for mean
    if full_models is None:
        with open(f'output/models/full_models_{timestamp}.pkl', 'rb') as file:
            full_models = pkl.load(file)

    # read data
    if train is None:
        train = pd.read_csv(
            f'data/abt/{data_version}/train.csv', parse_dates=[0], index_col=[0]
        )

    # Divide into target and feature matrices
    y_train, X_train = prepare_data_for_forecasting_model(train)

    # Replace anomalies with np.nan
    pipe_validate = full_models[list(full_models.keys())[0]].named_steps['validator']
    train_validated = pipe_validate.transform(X_train)

    # Ensure missing values are imputed first
    pipe_impute = full_models[list(full_models.keys())[0]].named_steps['imputer']
    X_train = pipe_impute.transform(train_validated)

    # Make quantile predictions for each model
    quantile_models = dict()
    for i, (model_name, model) in enumerate(full_models.items()):

        if testing is True:
            if i > 0:
                break

        logging.info('Computing quantiles for', i, model_name)

        # Set save folder names
        model_save_name = create_model_save_name(model_name)

        X_train_transformed = Pipeline(model.named_steps['estimator'].steps[:-1]).transform(X_train)

        lm = model.named_steps['estimator'].steps[-1][-1]

        SF = SelectFromModel(lm, max_features=100)

        # fit models
        quantile_models[model_name] = dict()
        if testing is False:
            quantiles = [0.05, 0.25, 0.75, 0.95]
        else:
            quantiles = [0.05]

        for q in quantiles:

            print('Fit', q)
            mqr = MultiOutputRegressor(LinearQuantileRegressor(quantiles=[q], tol=1e-2))

            try:
                pipeline = Pipeline([('select', SF), ('qr', mqr)])
                pipeline.fit(X_train_transformed, y_train)
            except:
                pipeline = Pipeline([('qr', mqr)])
                pipeline.fit(X_train_transformed, y_train)

            if testing is False:
                Path(f'output/qmodels/{model_save_name}/').mkdir(parents=True, exist_ok=True)
                with open(f'output/qmodels/{model_save_name}/quantiles_{q}_{timestamp}.pkl', 'wb') as file:
                    pkl.dump(pipeline, file)

            quantile_models[model_name][q] = pipeline

    return quantile_models


def assess_model_performances(test, full_models):
    """Given a test dataset and a dictionary of trained model objects,
    return a dataframe of r2 scores for the average next 24h that we
    forecast.

    Parameters
    ----------
    test : pd.DataFrame
        Test data
    full_models : dict
        Contains full model objects.

    Returns
    -------
    r2_scores : pd.DataFrame
        r2_scores for the next 24h of forecasted values for each model
    """

    # Assess model performance
    y_test, X_test = prepare_data_for_forecasting_model(test)

    r2_scores = dict()
    for model_name, model in full_models.items():

        # transform all input data
        imputer = Pipeline([
            ('ift', model.named_steps['imputer'].named_steps['ift']),
            ('it', model.named_steps['imputer'].named_steps['it']),
            ('soft', SelectOriginalFeaturesTransformer(X_test)),  # important to use input data
        ])
        preproc_pipeline = Pipeline([
            ('imputer', imputer),
            ('forecaster', model.named_steps['estimator'][:-1])
        ])
        X_test_transform = preproc_pipeline.transform(X_test)

        pred_test = pd.DataFrame(
            model.named_steps['estimator'].named_steps['lm'].predict(X_test_transform),
            index=y_test.index,
            columns=y_test.columns
        )

        scores = pd.concat([
            pd.DataFrame(
                calculate_metrics(np.array(y_test[col]), np.array(pred_test[col])), index=[0]
            )
            for col in y_test.columns
        ]).reset_index(drop=True)
        r2_scores[model_name] = scores['r2']

        if () in r2_scores.keys():
            r2_scores['full model'] = r2_scores[()]
            del r2_scores[()]

    # Add baseline model (simply taking the last known value)
    for lag, col in enumerate(y_test.columns):
        pred_test.loc[:, col] = y_test.loc[:, col].shift(lag + 1)
    scores = pd.concat([
        pd.DataFrame(
            calculate_metrics(np.array(y_test[col]), np.array(pred_test[col])), index=[0]
        )
        for col in y_test.columns
    ]).reset_index(drop=True)
    r2_scores['benchmark'] = scores['r2']

    return pd.DataFrame(r2_scores)
