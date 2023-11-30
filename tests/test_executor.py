import logging
from pathlib import Path

from darrow_poc.models.poc import POCAnomaly
from darrow_poc.mocks import ExecutorMock

import unittest


logging.basicConfig(level=logging.INFO)
logging.getLogger("sam").setLevel(logging.WARNING)


BASE_DIR = Path(__file__).parent.parent


class TestModelWithLocalExecutor(unittest.TestCase):
    def test_model_with_local_executor(self):
        config = {
            "model": POCAnomaly,
            "train_data_path": BASE_DIR / "tests/testing_data/train.parquet",
            "prediction_data_path": BASE_DIR / "tests/testing_data/test.parquet",
            "model_path": BASE_DIR / "output/models",
            "model_name": "poc_model",
            "predictions_path": BASE_DIR / "output/predictions/predictions.parquet",
        }
        executor = ExecutorMock(config)
        executor.run_full_flow()
        assert True


if __name__ == "__main__":
    unittest.main()
