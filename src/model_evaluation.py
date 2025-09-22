import logging
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, mean_squared_error


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model, X_test, y_test) -> dict:
        """_summary_

        Args:
            X_test (_type_): _description_
            y_test (_type_): _description_

        Returns:
            dict: _description_
        """
        pass


class RegressionLinearEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """_summary_

        Args:
            model (_type_): _description_
            X_test (_type_): _description_
            y_test (_type_): _description_
        """
        # Ensure the inputs are the correct type
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")
        if not isinstance(y_test, pd.DataFrame):
            raise TypeError("y_test must be a pandas Series or DataFrame.")
        
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)
        
        logging.info("Calculating evaluation metrics.")
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        metrics = {"Mean Squared Error": mse, "R-Squared": r2}
        logging.info(F"Model evaluation metrics: {metrics}.")
        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """_summary_

        Args:
            strategy (ModelEvaluationStrategy): _description_
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """_summary_

        Args:
            strategy (ModelEvaluationStrategy): _description_
        """
        self._strategy = strategy
    
    def evaluate(self, model, X_test, y_test):
        """_summary_

        Args:
            X_test (_type_): _description_
            y_test (_type_): _description_
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate(model, X_test, y_test)


if __name__ == "__main__":
    pass