import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod


class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            features (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        pass


class ZScoreOutlierDetectionStrategy(OutlierDetectionStrategy):
    def __init__(self, threshold=None):
        """_summary_

        Args:
            threshold (_type_, optional): _description_. Defaults to None.
        """
        self.threshold = threshold
    
    def detect(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        logging.info("Detecting outlier using Z-Score method.")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outlier detected with Z-Score threshold: {self.threshold}")
        return outliers


class IQROutlierDetectionStrategy(OutlierDetectionStrategy):
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        logging.info("Detecting outlier using IQR method.")
        lower = df.quantile(0.25)
        upper = df.quantile(0.75)
        IQR = upper - lower
        outlier = (df > (IQR + 1.5 * upper)) | (df < (IQR - 1.5 * lower))
        logging.info("Outlier detected with IQR method.")
        return outlier.any().any()


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        """_summary_

        Args:
            strategy (OutlierDetectionStrategy): _description_
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: OutlierDetectionStrategy):
        """_summary_

        Args:
            strategy (OutlierDetectionStrategy): _description_
        """
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        logging.info("Detecting outlier with {self._strategy}.")
        return self._strategy.detect(df)
    
    def handle(self, df: pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            method (str, optional): _description_. Defaults to "remove".

        Returns:
            pd.DataFrame: _description_
        """
        outlier = self._strategy.detect(df)
        if method == "remove":
            logging.info("Removing outliers from dataset.")
            df_cleaned = df[(~outlier).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df   

        logging.info("Outlier handling completed.")
        return df_cleaned


if __name__ == "__main__":
    pass