import logging
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression


class ModelTrainingStrategy(ABC):
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train) -> RegressorMixin:
        """_summary_

        Args:
            X_train (pd.DataFrame): _description_
            y_train (_type_): _description_

        Returns:
            RegressorMixin: _description_
        """
        pass


class LinearRegressionStrategy(ModelTrainingStrategy):
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        # ensure the inputs are the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.DataFrame):
            raise TypeError("y_train must be a pandas Series or DataFrame.")
        
        logging.info("Initializing linear regression model...")
        model = LinearRegression()
        
        logging.info("Training linear regression model.")
        model.fit(X_train, y_train)
        
        logging.info("Training model completed.")
        return model

class ModelTrainer:
    def __init__(self, strategy: ModelTrainingStrategy):
        """_summary_

        Args:
            strategy (ModelTrainingStrategy): _description_
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelTrainingStrategy):
        """_summary_

        Args:
            strategy (ModelTrainingStrategy): _description_
        """
        self._strategy = strategy
    
    def train(self, X_train, y_train):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        logging.info("Training model using the current strategy.")
        return self._strategy.train(X_train, y_train)


if __name__ == "__main__":
    pass