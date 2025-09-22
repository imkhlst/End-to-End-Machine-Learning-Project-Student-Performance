import logging
import pandas as pd


class IngestData:
    """
    Ingesting the data from the file_path.
    """
    def __init__(self, file_path: str):
        """
        Initializes the IngestData.

        Args:
            file_path (str): Path to the data.
        """
        self.file_path =file_path
    
    def get_data(self) -> pd.DataFrame:
        """
        Ingesting the data from the file_path.

        Returns:
            pd.DataFrame: A DataFrame with ingested data.
        """
        logging.info(f"Ingesting data from {self.file_path}.")
        return pd.read_csv(self.file_path)


if __name__ == "__main__":
    pass