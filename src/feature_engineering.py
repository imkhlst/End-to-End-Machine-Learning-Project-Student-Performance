import logging
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        pass


class LogTransformationStrategy(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        """_summary_

        Args:
            features (list): _description_
        """
        self.features = features
        
    def transform(self, df: pd.DataFrame, features: list):
        """_summary_

        Args:
            df (_type_): _description_
            features (_type_): _description_
        """
        logging.info("Applying log transformation to features: {self.features}.")
        df_transformed = df.copy()
        for feature in features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed


class StandardScalingStrategy(FeatureEngineeringStrategy):
    def __init__(self, features: list):
        """_summary_

        Args:
            features (list): _description_
        """
        self.features = features
        self.scaler = StandardScaler()
    
    def transform(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            features (list): _description_
        """
        logging.info("Applying Standard scaler to features: {self.features}.")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


class MinMaxScalingStrategy(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """_summary_

        Args:
            features (_type_): _description_
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def transform(self, df):
        """_summary_

        Args:
            df (_type_): _description_
        """
        logging.info("Applying Min-Max scaler to features: {self.features} with range: {self.scaler.feature_range}.")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


class LabelEncodingStrategy(FeatureEngineeringStrategy):
    def __init__(self, features):
        """_summary_

        Args:
            features (_type_): _description_
        """
        self.features = features
        self.encoder = LabelEncoder()
    
    def transform(self, df):
        """_summary_

        Args:
            df (_type_): _description_
        """
        logging.info("Applying label encoder to features: {self.features}.")
        df_transformed = df.copy()
        for features in self.features:
            df_transformed[features] = self.encoder.fit_transform(df[features])
        logging.info("Label encoding completed.")
        return df_transformed


class OneHotEncodingStrategy(FeatureEngineeringStrategy):
    def __init__(self, features):
        """_summary_

        Args:
            features (_type_): _description_
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")
    
    def transform(self, df):
        """_summary_

        Args:
            df (_type_): _description_
        """
        logging.info("Applying One-Hot encoder to features: {self.features}.")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-Hot encoding completed.")
        return df_transformed


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """_summary_

        Args:
            strategy (FeatureEngineeringStrategy): _description_
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """_summary_

        Args:
            strategy (FeatureEngineeringStrategy): _description_
        """
        self._strategy = strategy
    
    def apply(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): _description_
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.transform(df)



if __name__ == "__main__":
    pass