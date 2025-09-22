import logging
import pandas as pd

from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame, target_column: str):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            target_column (str): _description_
        """
        pass


class SimpleTrainTestSplit(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        """_summary_

        Args:
            test_size (float): _description_. Defaults to 0.2.
            random_state (int): _description_. Defaults to 42.
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, df, target_column):
        """_summary_

        Args:
            df (_type_): _description_
            target_column (_type_): _description_
        """
        logging.info("Perform simplet train-test split method.")
        X = df.drop(columns=target_column)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test


class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """_summary_

        Args:
            strategy (DataSplittingStrategy): _description_
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataSplittingStrategy):
        """_summary_

        Args:
            strategy (DataSplittingStrategy): _description_
        """
        self._strategy = strategy
    
    def split(self, df: pd.DataFrame, target_column: str):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            target_column (str): _description_
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split(df, target_column)


if __name__ == "__main__":
    pass