from networksecurity.constants import training_pipeline
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self, model, preprocessor):
        try:
            self.model = model
            self.preprocessor = preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def predict(self, x):
        try:
            x_transformed = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e, sys)