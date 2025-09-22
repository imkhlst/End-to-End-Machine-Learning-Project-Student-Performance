import logging
import pandas as pd
import numpy as np
import mlflow

from steps.data_ingestion_step import ingest
from steps.data_cleaning_step import clean
from steps.data_transforming_step import transform
from steps.data_splitting_step import split
from steps.model_training_step import train
from steps.model_evaluator_step import evaluate
from urllib.parse import urlparse


def train_pipeline(file_path: str):
    try:
        df = ingest(file_path)
        df_cleaned = clean(df)
        numerical_features = df_cleaned.select_dtypes(include=np.number).columns
        categorical_features = df_cleaned.select_dtypes(include="O").columns
        df_transformed = transform(df_cleaned, strategy="label_encoding", features=categorical_features)
        X_train, X_test, y_train, y_test = split(df_transformed, numerical_features)
              
        logging.info("Training and Evaluation with linear regression model.")
        trained_model = train(X_train, y_train)
        
        logging.info("Training Completed.")
        evaluation_metrics  = evaluate(trained_model, X_test, y_test)
        
        logging.info("Evaluation completed.")
        
        mlflow.log_metric("MSE", evaluation_metrics["Mean Squared Error"])
        mlflow.log_metric("R2", evaluation_metrics["R-Squared"])
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(trained_model, "model", registered_model_name="LinearRegressionModel")
        else:
            mlflow.sklearn.log_model(trained_model, "model")
            
        return evaluation_metrics

    except Exception as e:
        logging.warning(f"Error occur while running training pipeline: {e}")
        raise e