import logging
import pandas as pd
from typing import Tuple

from src.data_splitting import DataSplitter, SimpleTrainTestSplit


def split(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        splitter = DataSplitter(SimpleTrainTestSplit())
        X_train, X_test, y_train, y_test = splitter.split(df, target_column)
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.warning(f"Error occur while spltting data: {e}.")
        raise e