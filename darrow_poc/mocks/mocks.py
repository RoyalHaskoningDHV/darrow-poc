import os
import pandas as pd
from typing import Any

from dataclasses import dataclass

from twinn_ml_interface.input_data import InputData
from twinn_ml_interface.objectmodels import (
    Configuration,
    MetaDataLogger,
    RelativeType,
    Unit,
    UnitTag,
    UnitTagTemplate,
)
from twinn_ml_interface.interface import ModelInterfaceV4


@dataclass
class LocalConfig:
    """Class for configuring model adhering to ModelInterfaceV4"""
    model: ModelInterfaceV4
    train_data_path: os.PathLike
    prediction_data_path: os.PathLike
    model_path: os.PathLike
    model_name: str
    predictions_path: os.PathLike = "/my/path/predictions/predictions.parquet"


class InfraConfigurationMock(Configuration):
    """Use to get information from hierarchy (rooted tree) and tenant.

    Not that important in this mock setting. We only need the `target_name`.
    The rest is shown here as well for completeness. Not something you need
    to implement, but something you can use. Note that for now we are not mocking
    any of the output except `target_name`.
    """

    target_name = "stah:discharge"

    def tenant(self) -> dict[str, Any]:
        return

    def tenant_config(self) -> dict[str, Any]:
        """Get and cache the tenant_config."""
        return None

    def get_unit_properties(self, unit_name: str) -> dict[str, Any] | None:
        """Retrieve the property of a certain unit.

        Args:
            unit_name (str): name of the unit to get properties for.

        Returns:
            dict[str, Any] | None: the property of the UnitTag if it exists.
        """
        return None

    def get_units(self, unit_name: str, relative_path: list[RelativeType]) -> list[Unit] | None:
        """Retrieve units from the hierarchy.

        Args:
            unit_name (str): name of the unit to search from.
            relative_path (list[RelativeType]): a path to search for relative to the given unit.

        Returns:
            list[Unit] | None: the units.
        """
        return None

    def get_unit_tags(self, unit_name: str, unit_tag_template: UnitTagTemplate) -> list[UnitTag]:
        """Retrieve UnitTags from the hierarchy.

        Args:
            unit_name (str): name of the unit to search from.
            unit_tag_template (UnitTagTemplate): a relative path from the given unit.

        Returns:
            list[UnitTag]: the UnitTags that were found.
                You can easily convert them to strings by calling str() on them.
        """
        return None


class ExecutorMock:
    """A mock executor, that mimics some of the behaviour of a real executor.

    An executor is responsible for running a machine learning model during training
    or predicting based on a `ModelInterfaceV4` compliant class in the `darrow-ml-platform`.

    This mock executor performs both training and predicting, but only locally. Not all
    aspects of a real executor are mocked, but the basic logic flow should be the same.
    """

    metadata_logger = MetaDataLogger()

    def __init__(self, local_config: LocalConfig):
        self.local_config = local_config

    def _init_train(self) -> tuple[ModelInterfaceV4, Configuration]:
        model_class = self.local_config.model
        infra_config = InfraConfigurationMock()
        return model_class, infra_config

    def get_training_data(
        self,
        model: ModelInterfaceV4,
        infra_config: Configuration | None = None,
    ) -> InputData:
        """Get training data input for ML model

        Args:
            model (ModelInterfaceV4): ML model
            infra_config (Configuration | None, optional): Used to get info from hierarchy and tenant

        Returns:
            InputData: Input data for ML model
        """
        long_data = pd.read_parquet(self.local_config.train_data_path)
        return InputData.from_long_df(long_data)

    def _write_model(self, model: ModelInterfaceV4) -> None:
        # When running the model in our infra, we store all the logs and then we reset the
        # cache before dumping the model. This means that MetaDataLogger contents won't be available
        # when loading the model for predictions
        self.metadata_logger.reset_cache()
        model.dump(self.local_config.model_path, self.local_config.model_name)

    def _postprocess_model_results(self, model: ModelInterfaceV4):
        model.base_features = None
        model.performance_value = None

    def write_results(self, model: ModelInterfaceV4):
        """Save model to file.

        Args:
            model (ModelInterfaceV4): _description_
        """
        self._write_model(model=model)
        self._postprocess_model_results(model=model)

    def run_train_flow(self):
        """Run training flow and cache trained model"""
        model_class, infra_config = self._init_train()
        model = model_class.initialize(infra_config, self.metadata_logger)

        input_data = self.get_training_data(model, infra_config)
        preprocessed_data = model.preprocess(input_data)
        model.train(preprocessed_data, save_performance=True)  # TODO (Team): discuss what save_performance stands for

        self.write_results(model)

    def load_model(self, model_class):
        """Load saved ML model

        Args:
            model_class (ModelInterfaceV4): Name of the model class

        Returns:
            ModelInterfaceV4: ML model
        """
        return model_class.load(self.local_config.model_path, self.local_config.model_name)

    def get_prediction_data(self) -> InputData:
        """Get input data for predicting

        Returns:
            InputData: Input data for ML model
        """
        long_data = pd.read_parquet(self.local_config.prediction_data_path)
        return InputData.from_long_df(long_data)

    def write_predictions(self, predictions: pd.DataFrame):
        """Write predictions to local path. When running the actual infrastructure,
        predictions are uploaded to the azure data lake.

        Args:
            predictions (pd.DataFrame): Predictions made by ML Model
        """
        predictions.to_parquet(self.local_config.predictions_path)

    def run_predict_flow(self):
        """Run predict flow"""
        model: ModelInterfaceV4 = self.load_model(self.local_config.model)

        input_data = self.get_prediction_data()
        preprocessed_data = model.preprocess(input_data)
        predictions, _ = model.predict(preprocessed_data)
        self.write_predictions(pd.concat(predictions))

    def run_full_flow(self):
        """Run both train and predict flows"""
        self.run_train_flow()
        self.run_predict_flow()
