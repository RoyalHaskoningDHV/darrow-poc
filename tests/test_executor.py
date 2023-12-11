import logging
from pathlib import Path

from darrow_poc.models.poc import POCAnomaly
from darrow_poc.mocks import ModelConfig, ExecutorMock

import unittest


logging.basicConfig(level=logging.INFO)
logging.getLogger("sam").setLevel(logging.WARNING)


BASE_DIR = Path(__file__).parent.parent


class TestModelWithLocalExecutor(unittest.TestCase):
    def test_model_with_local_executor(self):
        config = ModelConfig(
            POCAnomaly,
            BASE_DIR / "tests/testing_data/train.parquet",
            BASE_DIR / "tests/testing_data/test.parquet",
            BASE_DIR / "output/models",
            "poc_model",
            BASE_DIR / "output/predictions/predictions.parquet",
        )
        executor = ExecutorMock(config)
        executor.run_full_flow()
        assert True


if __name__ == "__main__":
    unittest.main()
