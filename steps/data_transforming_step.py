import logging
import pandas as pd
import numpy as np

from src.feature_engineering import (
    FeatureEngineer,
    LabelEncodingStrategy,
    StandardScalingStrategy,
    LogTransformationStrategy,
    MinMaxScalingStrategy,
    OneHotEncodingStrategy
)

def transform(df: pd.DataFrame, strategy="log", features: list=None) -> pd.DataFrame:
    try:
        # Ensure the features
        if features is None:
            raise ValueError("Features are {Features}. Input specific list of features.")
        
        if strategy == "log":
            engineer = FeatureEngineer(LogTransformationStrategy(features))
        elif strategy == "standard_scaling":
            engineer = FeatureEngineer(StandardScalingStrategy(features))
        elif strategy == "min_max_scaling":
            engineer = FeatureEngineer(MinMaxScalingStrategy(features))
        elif strategy == "label_encoding":
            engineer = FeatureEngineer(LabelEncodingStrategy(features))
        elif strategy == "one_hot_encoding":
            engineer = FeatureEngineer(OneHotEncodingStrategy(features))
        else:
            raise ValueError(f"Unsupported feature engineering strategy: {strategy}.")
        
        df_transformed = engineer.apply(df)
        return df_transformed
    
    except Exception as e:
        logging.warning("Error occur while transforming data: {e}.")
        raise e