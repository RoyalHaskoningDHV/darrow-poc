from twinn_ml_interface.interface import ModelInterfaceV4
from twinn_ml_interface.objectmodels import MetaDataLogger

from darrow_poc.models.modelinterface import POCAnomaly
from darrow_poc.mocks import ConfigurationMock
import unittest

import logging


logging.basicConfig(level=logging.DEBUG)


class TestModelFollowsInterface(unittest.TestCase):

    def test_model_follows_interface(self):
        sm = POCAnomaly.initialize(ConfigurationMock(), MetaDataLogger())
        assert isinstance(sm, ModelInterfaceV4)

if __name__ == "__main__":
    unittest.main()