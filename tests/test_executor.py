from pathlib import Path

from darrow_poc.models.modelinterface import POCAnomaly
from darrow_poc.mocks import ExecutorMock

import unittest

import logging


BASE_DIR = Path("~/projects/darrow/darrow-poc/")


logging.basicConfig(level=logging.DEBUG)


class TestModelWithLocalExecutor(unittest.TestCase):

    def test_model_with_local_executor(self):
        config = {
            "model": POCAnomaly,
            "train_data_path": BASE_DIR / "tests/testing_data/train.parquet",
            "model_path": BASE_DIR / "tests/testing_models",
            "model_name": "poc_model",
            "predictions_path": BASE_DIR / "predictions/predictions.parquet",
        }
        executor = ExecutorMock(config)
        executor.run_full_flow()
        assert True


if __name__ == "__main__":
    unittest.main()
