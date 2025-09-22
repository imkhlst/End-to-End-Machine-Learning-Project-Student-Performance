import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.impute import SimpleImputer 


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing value in the DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame containing features with missing values.
        
        Returns:
            pd.DataFrame: A DataFrame with missing values handled features.
        """
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, threshold=None):
        """
        Initializes the DropMissingValuesStrategy with specific parameter.

        Args:
            axis (int): 0 to drops rows with the missing values, 1 to drop columns with missing values.
            threshold (int): the Threshold for non-NA values. rows/columns with less than threshold non-NA values are dropped.
        
        Returns:
            None
        """
        self.axis = axis
        self.threshold = threshold
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing value based on the axis and the threshold.

        Args:
            df (pd.DataFrame): A DataFrame containing features with missing values.

        Returns:
            pd.DataFrame: A DataFrame with missing value handled data.
        """
        logging.info(f"Dropping missing value with axis={self.axis} and thresh={self.threshold}")
        df_cleaned = df.dropna(self.axis, thresh=self.threshold)
        logging.info("Missing values dropped.")
        return df_cleaned


class FillMissingValueStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValue with specific parameter.

        Args:
            method (str): _description_. Defaults to "mean".
            fill_value (any): _description_. Defaults to None.
        """
        self.method = method
        self.fill_value = fill_value
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing value using a specified method or constant value.

        Args:
            df (pd.DataFrame): A DataFrame containing features with missing values.

        Returns:
            pd.DataFrame: A DataFrame with missing value handled data.
        """
        logging.info(f"Filling missing values using method: {self.method}")
        
        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_features = df_cleaned.select_dtypes(include=np.number).columns
            df_cleaned[numeric_features] = df_cleaned[numeric_features].fillna(df[numeric_features].mean())
        elif self.method == "median":
            numeric_features = df_cleaned.select_dtypes(include=np.number).columns
            df_cleaned[numeric_features] = df_cleaned[numeric_features].fillna(df[numeric_features].median())
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[numeric_features].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.info(f"warning method '{self.method}'. No missing value handled.")
        
        logging.info("Missing values filled.")
        return df_cleaned


class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initializes the MissingValueHandler with a specific missing values handling strategy.

        Args:
            strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing value.
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy to handling missing values.

        Args:
            strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing value.
        """
        self._strategy = strategy
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes MissingValueHandler using the current strategy.

        Args:
            df (pd.DataFrame): A DataFrame containing features with missing values.

        Returns:
            pd.DataFrame: A DataFrame with missing value handled data.
        """
        logging.info("Executing missing value handler with '{self._strategy}'.")
        return self._strategy.handle(df)


if __name__ == "__main__":
    pass