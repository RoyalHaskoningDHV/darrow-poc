from __future__ import annotations
from os import PathLike
from pathlib import Path
from typing import Callable

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from twinn_ml_interface.interface import ModelInterfaceV4
from twinn_ml_interface.objectmodels import (
    AvailabilityLevel,
    Configuration,
    DataLabelConfigTemplate,
    DataLevel,
    InputData,
    MetaDataLogger,
    ModelCategory,
    RelativeType,
    Tag,
    UnitTagTemplate,
    UnitTag,
    Unit,
    WindowViability,
)

from .models import (
    train_validator,
    train_imputer,
    train_forecasting_models,
)
from .feature_engineering import (
    SelectOriginalFeaturesTransformer,
)


class POCAnomaly:

    model_type_name: str = "pocanomaly"
    # Model category is based on the output of the model.
    model_category: ModelCategory = ModelCategory.ANOMALY
    # Number between (-inf, inf) indicating the model performance.
    performance_value: float | None = None
    # Features used to train the model. If not supplied, equal to get_data_config_template().
    base_features: dict[DataLevel, list[UnitTag]] | None = None
    # This is only needed when get_target_template returns UnitTagTemplate
    target: UnitTag | None = None

    def __init__(self, target: UnitTag):
        self.target = target

    @staticmethod
    def get_target_template() -> UnitTagTemplate | UnitTag:
        """Get the name of the target tag to train the model.

        Returns:
            UnitTagTemplate | UnitTag: The unit tag of the model target, either as template or as UnitTag.
        """
        return UnitTag(Unit("STAHROER", "DISCHARGE_STATION", True), Tag("DISCHARGE"))

    @staticmethod
    def get_data_config_template() -> list[DataLabelConfigTemplate] | list[UnitTag]:
        """The specification of data needed to train and predict with the model.

        Result:
            list[DataLabelConfigTemplate] | list[UnitTag]: The data needed to train and predict with the model,
                either as template or as list of literals.
        """
        return [
            DataLabelConfigTemplate(
                data_level=DataLevel.SENSOR,
                unit_tag_templates=[
                    UnitTagTemplate([RelativeType.CHILDREN], [Tag("DISCHARGE")])
                ],
                availability_level=AvailabilityLevel.available_until_now,
            ),
            DataLabelConfigTemplate(
                data_level=DataLevel.WEATHER,
                unit_tag_templates=[
                    UnitTagTemplate([RelativeType.CHILDREN], [Tag("PRECIPITATION"), Tag("EVAPORATION")])
                ],
                availability_level=AvailabilityLevel.available_until_now,
            ),
        ]

    @staticmethod
    def get_result_template() -> UnitTagTemplate | UnitTag:
        """The tag to post the predictions/results on.

        Returns:
           UnitTagTemplate, UnitTag: The unit tag of the model's output, either as template or as literal.
        """
        return UnitTag(Unit("STAHROER", "DISCHARGE_STATION", True), Tag("DISCHARGE_FORECAST"))

    @staticmethod
    def get_unit_properties_template() -> list[Tag]:
        """Unit properties to get from the units specified in data_config.

        Returns:
            list[Tag]: The tags to request.
        """
        return []

    @staticmethod
    def get_unit_hierarchy_template() -> dict[str, list[RelativeType]]:
        """Request some units from the hierarchy in a dictionary.

        Returns:
            dict[str, list[RelativeType]]: An identifier for the units to get, and their relative path from the target unit.
        """
        return {}

    @staticmethod
    def get_train_window_finder_config_template() -> list[DataLabelConfigTemplate] | None:
        """The config for running the train window finder.

        Returns:
            list[DataLabelConfigTemplate] | None: a template for getting the tags needed to run the train window
                finder. Defaults to None, then no train window finder will be used.
        """
        return None

    @classmethod
    def initialize(cls, configuration: Configuration, logger: MetaDataLogger) -> ModelInterfaceV4:
        """Post init function to pass metadata logger and some config to the model.

        NOTE:
        This is used, because we cannot inherit an __init__() from the Protocol, and because
        passing configuration and logger to each method where they are needed would be a little
        tedious.

        Args:
            logger (MetaDataLogger): A MetaDataLogger object to write logs to MLflow later.
            tenant_config (dict[str, Any]): Tenant specific configuration.
        """
        stahmodel = cls(configuration.target_name)
        stahmodel.configuration = configuration
        stahmodel.logger = logger
        return stahmodel

    def preprocess(self, input_data: InputData) -> InputData:
        """Preprocess input data before training.

        Args:
            data (InputData): Input data.

        Returns:
            InputData: Preprocessed input data.

        """
        return input_data

    def validate_input_data(
        self,
        input_data: InputData,
    ) -> WindowViability:
        """Validate if input data is usable for training.

        Args:
            data (InputData): Training data.

        Returns:
            WindowViability: For each PredictionType you get
                bool: Whether the data can be used for training. Default always true.
                str: Additional information about the window.
        """
        return True, "Input data is valid."

    def train(self, input_data: InputData, **kwargs) -> None:
        """Train a model.

        Args:
            input_data (InputData): Preprocessed and validated training data.

        Returns:
            dict[str, Any] | None: Optionally some logs collected during training.
        """
        # Currently not dividing into train and test sets to speed things up
        train = pd.concat(input_data.values(), axis=1)
        validator, timestamp_validator = train_validator(
            train,
            n_features=5,
            model_type='lasso',
            testing=False,
            use_precipitation_features=False,
        )
        imputer, timestamp_imputer = train_imputer(train)
        full_models = train_forecasting_models(
            train, validator, imputer, timestamp=timestamp_validator, model_type='ridge'
        )
        self._full_models = full_models

    def predict(self, input_data: InputData, **kwargs) -> list[pd.DataFrame]:
        """Run a prediction with a trained model.

        Args:
            input_data (InputData): Prediction data.

        Returns:
            list[pd.DataFrame]: List of dataframes with predictions
        """
        model = self._full_models
        X = pd.concat(input_data.values(), axis=1)

        discharge_removed_anomalies = model.named_steps['validator'].transform(X.iloc[-24:, :])
        X_removed_anomalies = X.copy()
        X_removed_anomalies = pd.concat(
            [X_removed_anomalies.iloc[:-24, :], discharge_removed_anomalies]
        )

        # transform all input data
        imputer = Pipeline([
            ('ift', model.named_steps['imputer'].named_steps['ift']),
            ('it', model.named_steps['imputer'].named_steps['it']),
            ('soft', SelectOriginalFeaturesTransformer(
                X_removed_anomalies
            )),  # important to use input data
        ])
        preproc_pipeline = Pipeline([
            ('validator', model.named_steps['validator']),
            ('imputer', imputer),
            ('forecaster', model.named_steps['estimator'][:-1])
        ])
        X_transformed = preproc_pipeline.transform(X_removed_anomalies)

        # select last complete sample
        X_transformed = X_transformed[~np.isnan(X_transformed).any(axis=1)]

        # create predictions
        pred = model.named_steps['estimator'][-1].predict(X_transformed[-1:, :]).flatten()

        return pred  # [pd.Dataframe({'A': [1, 2, 3]})]

    def dump(self, foldername: PathLike, filename: str) -> None:
        """
        Writes the following files:
        * filename.pkl
        * filename.h5
        to the folder given by foldername.

        Args:
            foldername (PathLike): configurable folder name
            filename (str): name of the file
        """
        with open(Path(foldername) / (filename + ".pkl"), "wb") as f:
            pickle.dump(self, f)
        return None

    @classmethod
    def load(cls, foldername: PathLike, filename: str) -> Callable:
        """
        Reads the following files:
        * filename.pkl
        * filename.h5
        from the folder given by foldername.
        Output is an entire instance of the fitted model that was saved

        Args:
            foldername (PathLike): configurable folder name
            filename (str): name of the file

        Returns:
            Model class with everything (except data) contained within to call the
            `predict()` method
        """
        with open(Path(foldername) / (filename + ".pkl"), "rb") as f:
            model = pickle.load(f)

        return model
