import pandas as pd

from darrow_poc.models import POCAnomaly

from twinn_ml_interface.input_data import InputData
from twinn_ml_interface.objectmodels import MetaDataLogger, Configuration
from twinn_ml_interface.interface import ModelInterfaceV4


# TODO (Steffen): Maybe missing info, test
CONFIG = {
    "model": POCAnomaly,
    "data_path": "/my/path/data/data.parquet",
    "model_path": "/my/path/model/",
    "model_name": "my_model",
    "predictions_path": "/my/path/predictions/predictions.parquet",
}


# TODO: Better config, with maybe some methods that return something?
class ConfigurationMock:
    target_name = "test:test"

    def get_units(*args, **kwargs):
        return None


# TODO: Add methods where we access the model attributes, like `model_category`
class ExecutorMock:
    metadata_logger = MetaDataLogger()

    def __init__(self, config: dict = CONFIG):
        self.config = config

    def _test_model_attributes(self, model_class):
        # The isinstance check with the annotation protocol already does this,
        # but just to be super clear that the actual exectutors expect these
        return True
        assert hasattr(self, "model_type_name"), "'model_type_name' attribute is missing!"
        assert hasattr(self, "model_category"), "'model_category' attribute is missing!"
        assert hasattr(self, "performance_value"), "'performance_value' attribute is missing!"
        assert hasattr(self, "base_features"), "'base_features' attribute is missing!"
        assert hasattr(self, "target"), "'target' attribute is missing!"

    def init_train(self, str) -> tuple[ModelInterfaceV4, Configuration]:
        model_class = self.config["model"]
        config_api = ConfigurationMock()
        return model_class, config_api

    def get_training_data(
        self,
        model: ModelInterfaceV4,
        config_api: Configuration | None = None,
    ) -> InputData:
        long_data = pd.read_parquet(self.config["train_data_path"])
        return InputData.from_long_df(long_data)

    def _write_model(self, model: ModelInterfaceV4) -> None:
        # When running the model in our infra, we store all the logs and then we reset the
        # cache before dumping the model. This means that MetaDataLogger contents won't be available
        # when loading the model for predictions
        self.metadata_logger.reset_cache()
        model.dump(self.config["model_path"], self.config["model_name"])

    def postprocess_model_results(self, model: ModelInterfaceV4):
        model.base_features = None
        model.performance_value = None

    def write_results(self, model: ModelInterfaceV4):
        self._write_model(model=model)
        self.postprocess_model_results(model=model)

    def run_train_flow(self):
        model_class, config_api = self.init_train(self.config)
        self._test_model_attributes(model_class)
        model = model_class.initialize(config_api, self.metadata_logger)

        input_data = self.get_training_data(model, config_api)
        preprocessed_data = model.preprocess(input_data)
        model.train(preprocessed_data, save_performance=True)  # TODO (Team): discuss what save_performance stands for

        self.write_results(model)

    def load_model(self, model_class):
        return model_class.load(self.config["model_path"], self.config["model_name"])

    def get_prediction_data(
        self,
        model: ModelInterfaceV4,
    ) -> InputData:
        long_data = pd.read_parquet(self.config["prediction_data_path"])
        return InputData.from_long_df(long_data)

    def write_predictions(self, predictions: pd.DataFrame):
        """Write predictions to local path. When running the actual infrastructure,
        predictions are uploaded to the azure data lake.

        Parameters
        ----------
        predictions: pd.DataFrame
        """
        predictions.to_parquet(self.config["predictions_path"])

    def run_predict_flow(self):
        model: ModelInterfaceV4 = self.load_model(self.config["model"])

        input_data = self.get_prediction_data(model)
        preprocessed_data = model.preprocess(input_data)
        predictions = model.predict(preprocessed_data)
        self.write_predictions(predictions)

    def run_full_flow(self):
        self.run_train_flow()
        self.run_predict_flow()
