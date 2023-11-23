from twinn_ml_interface.interface import ModelInterfaceV4
from twinn_ml_interface.objectmodels import MetaDataLogger

from darrow_poc.models.modelinterface import POCAnomaly
import unittest

import logging


logging.basicConfig(level=logging.DEBUG)

class ConfigurationMock:
    target_name = "test:test"

    def get_units(*args, **kwargs):
        return None


class TestModelFollowsInterface(unittest.TestCase):

    def test_model_follows_interface(self):
        sm = POCAnomaly.initialize(ConfigurationMock(), MetaDataLogger())
        assert isinstance(sm, ModelInterfaceV4)

if __name__ == "__main__":
    unittest.main()