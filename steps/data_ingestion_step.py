import logging
from src.data_ingestion import IngestData

import pandas as pd


def ingest(file_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the file_path.

    Args:
        file_path (str): Path to the data.

    Returns:
        pd.DataFrame: A DataFrame with ingested data.
    """
    try:
        ingest = IngestData(file_path)
        df = ingest.get_data()
        return df
    except Exception as e:
        logging.warning(f" Error while ingesting data: {e}.")
        raise e