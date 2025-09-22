import logging
import pandas as pd
import numpy as np

from src.handling_missing_value import MissingValueHandler, FillMissingValueStrategy
from src.outlier_detection import OutlierDetector, IQROutlierDetectionStrategy

def clean(df: pd.DataFrame) -> pd.DataFrame:
    try:
        handler = MissingValueHandler(FillMissingValueStrategy(method="mean"))
        df_cleaned = handler.handle(df)
        numerical_features = df_cleaned.select_dtypes(include=np.number).columns
        detector = OutlierDetector(IQROutlierDetectionStrategy())
        outlier_found = detector.detect(df_cleaned[numerical_features])
        if outlier_found:
            df_cleaned[numerical_features] = detector.handle(df_cleaned[numerical_features], method="cap")
        else:
            logging.info("No outliers detected.")
        
        return df_cleaned
    except Exception as e:
        logging.warning(f"Error occur while cleaning data: {e}.")
        raise e