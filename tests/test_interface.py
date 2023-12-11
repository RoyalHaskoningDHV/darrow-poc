from twinn_ml_interface.interface import ModelInterfaceV4
from twinn_ml_interface.objectmodels import MetaDataLogger

from darrow_poc.models.poc import POCAnomaly
from darrow_poc.mocks import ConfigurationMock
import unittest

import logging


logging.basicConfig(level=logging.DEBUG)


class TestModelFollowsInterface(unittest.TestCase):
    def test_model_follows_interface(self):
        pocanomaly = POCAnomaly.initialize(ConfigurationMock(), MetaDataLogger())
        if not isinstance(pocanomaly, ModelInterfaceV4):
            raise TypeError(
                f"{pocanomaly.__name__} does not conform to the Protocol {ModelInterfaceV4.__name__}."
            )


if __name__ == "__main__":
    unittest.main()
