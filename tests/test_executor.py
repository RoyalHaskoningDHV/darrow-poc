import logging
from pathlib import Path

from darrow_poc.models.poc import POCAnomaly
from twinn_ml_interface.mocks import LocalConfig, ExecutorMock, ConfigurationMock

import unittest


logging.basicConfig(level=logging.INFO)
logging.getLogger("sam").setLevel(logging.WARNING)


BASE_DIR = Path(__file__).parent.parent


class TestModelWithLocalExecutor(unittest.TestCase):
    def test_model_with_local_executor(self):
        config = LocalConfig(
            POCAnomaly,
            BASE_DIR / "tests/testing_data/train.parquet",
            BASE_DIR / "tests/testing_data/test.parquet",
            BASE_DIR / "output/models",
            "poc_model",
            BASE_DIR / "output/predictions/predictions.parquet",
        )
        infra_config = ConfigurationMock("stah:discharge", "", {}, [], [])
        executor = ExecutorMock(config, infra_config)
        executor.run_full_flow()


if __name__ == "__main__":
    unittest.main()
