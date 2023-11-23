from twinn_ml_interface.interface import ModelInterfaceV4
from twinn_ml_interface.objectmodels import ModelCategory

from darrow_poc.models.modelinterface import POCAnomaly

import logging


logging.basicConfig(level=logging.DEBUG)


sm = POCAnomaly(target=None)
sm.performance_value = 999
sm.model_category = ModelCategory.ANOMALY
sm.model_type_name = "stah"
sm.base_features = None

assert isinstance(sm, ModelInterfaceV4)
