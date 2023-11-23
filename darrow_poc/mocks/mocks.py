from twinn_ml_interface.objectmodels import MetaDataLogger, Configuration, InputData
from twinn_ml_interface.interface import ModelInterfaceV4
import pandas as pd

from darrow_poc.models import POCAnomaly


# TODO (Steffen): Maybe missing info, test
CONFIG = {
    "model" : POCAnomaly,
    "data_path" : "/my/path/hello.parquet"
}

# TODO: Complete this function
def init_train(str) -> tuple[ModelInterfaceV4, Configuration]:
    return ""

# TODO: Better config, with maybe some methods that return something?
class ConfigurationMock: 
    target_name = "test:test"

    def get_units(*args, **kwargs):
        return None
    



# TODO: Add methods where we access the model attributes, like `model_category`
class ExecutorMock:

    metadata_logger = MetaDataLogger()

    def get_data(self,
        model: ModelInterfaceV4,
        target: str,
        config_api: Configuration,
    ) -> InputData:
        long_data = pd.read_parquet(CONFIG['data_path'])
        return InputData.from_long_df(long_data)
    
    def _write_model(self, model: ModelInterfaceV4) -> None:
        # When running the model in our infra, we store all the logs and then we reset the
        # cache before dumping the model. This means that MetaDataLogger contents won't be available
        # when loading the model for predictions
        self.metadata_logger.reset_cache()
        model.dump(CONFIG['model_path'], "my_model")
    
    def postprocess_model_results(self, model:ModelInterfaceV4):
        model.base_features
        model.performance_value

    def write_results(self, model: ModelInterfaceV4):
        self._write_model(model=model)
        self.postprocess_model_results(model=model)

    def run_train_flow(self): 
        model_class, config_api = init_train(CONFIG)
        model = model_class.initialize(configuration=config_api, logger=self.metadata_logger)

        input_data = self.get_data(
            model,
            config_api.target_name,
            config_api,
        )
        preprocessed_data = model.preprocess(input_data)
        model.train(preprocessed_data, save_performance=True) # TODO (Team): discuss what save_performance stands for

        self.write_results(model)

    def run_predict_flow(self):
        pass

    def run_full_flow(self):
        self.run_train_flow()
        self.run_predict_flow()