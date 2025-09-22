import logging
import pandas as pd

from src.model_training import ModelTrainer, LinearRegressionStrategy
from sklearn.base import RegressorMixin

def train(X_train, y_train) -> RegressorMixin:
    try:
        trainer = ModelTrainer(LinearRegressionStrategy())
        trained_model = trainer.train(X_train, y_train)
        return trained_model
    
    except Exception as e:
        logging.warning(f"Error occur while training model: {e}.")
        raise e