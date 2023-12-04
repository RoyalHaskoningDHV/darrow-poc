from __future__ import annotations
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import dill as pickle
import numpy as np
import pandas as pd

from twinn_ml_interface.input_data import InputData
from twinn_ml_interface.interface import ModelInterfaceV4
from twinn_ml_interface.objectmodels import (
    Configuration,
    DataLabelConfigTemplate,
    DataLevel,
    MetaDataLogger,
    ModelCategory,
    Tag,
    UnitTagTemplate,
    UnitTag,
    Unit,
    WindowViability,
)

from .anomaly_detection import ValidationModel, get_anomalies


class POCAnomaly:
    model_type_name: str = "pocanomaly"
    # Model category is based on the output of the model.
    model_category: ModelCategory = ModelCategory.ANOMALY
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
        return UnitTag(Unit("STAH", "DISCHARGE_STATION", True), Tag("DISCHARGE"))

    @staticmethod
    def get_data_config_template() -> list[DataLabelConfigTemplate] | list[UnitTag]:
        """The specification of data needed to train and predict with the model.

        NOTE:
        Using DataLabelConfigTemplate is more complicated, but allows refering to relative relationships
        between units (e.g. selecting all children of the target). Specifying UnitTags directly is simpler,
        but requires you to know the exact units necessary for the model.

        Result:
            list[DataLabelConfigTemplate] | list[UnitTag]: The data needed to train and predict with the model,
                either as template or as list of literals.
        """
        return [
            UnitTag.from_string("altenburg1:disc"),
            UnitTag.from_string("eschweiler:disc"),
            UnitTag.from_string("herzogenrath1:disc"),
            UnitTag.from_string("juelich:disc"),
            UnitTag.from_string("stah:disc"),
            UnitTag.from_string("evap:evap"),
            UnitTag.from_string("middenroer:prec"),
            UnitTag.from_string("urft:prec"),
        ]

    @staticmethod
    def get_result_template() -> UnitTagTemplate | UnitTag:
        """The tag to post the predictions/results on.

        Returns:
           UnitTagTemplate, UnitTag: The unit tag of the model's output, either as template or as literal.
        """
        return UnitTag(Unit("STAHROER", "DISCHARGE_STATION", True), Tag("DISCHARGE_FORECAST"))

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
        model = cls(configuration.target_name)
        model.configuration = configuration
        model.logger = logger
        return model

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

    def train(self, input_data: InputData, **kwargs) -> tuple[float, Any]:
        """Train a model.

        Args:
            input_data (InputData): Preprocessed and validated training data.

        Returns:
            float: Number between (-inf, inf) indicating the model performance
            Any: Any other object that can be used for testing. This object will be ignored
                by the infrastructure
        """
        train = pd.concat(input_data.values(), axis=1)
        validator = ValidationModel(
            train,
            model_type="lasso",
            n_features=5,
            use_precipitation_features=False,
            training_end_date="2010-01-04 00:00:00",
        )
        _, num_obs, _, r2_by_target = validator.fit_and_evaluate(str(self.target))
        r2_by_missing_sensor = validator.flatten_output(r2_by_target, "r2")
        self.logger.log_params(r2_by_missing_sensor)  # This will be logged to mlflow
        self.logger.log_params({f"samples_{k}": v for k, v in num_obs.items()})

        self._model = validator
        return np.mean(r2_by_missing_sensor.values()), None

    def predict(self, input_data: InputData, **kwargs) -> tuple[list[pd.DataFrame], Any]:
        """Run a prediction with a trained model.

        Args:
            input_data (InputData): Prediction data.

        Returns:
            list[pd.DataFrame]: List of dataframes with predictions
            Any: Any other object that can be used for testing. This object will be ignored
                by the infrastructure
        """
        target_channel = str(self.target)
        model = self._model
        X = pd.concat(input_data.values(), axis=1)
        predictions = model.predict(X, target_channel)
        anomalies = get_anomalies(predictions, X, target_channel)

        return [pd.DataFrame({target_channel: anomalies})], None

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
