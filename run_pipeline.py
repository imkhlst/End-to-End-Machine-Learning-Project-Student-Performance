import mlflow
import logging
from pipelines.training_pipeline import train_pipeline


if __name__ == "__main__":
    remote_server_uri = "http://ec2-13-212-243-180.ap-southeast-1.compute.amazonaws.com:5000/"
    mlflow.set_tracking_uri(remote_server_uri)
    
    if not mlflow.active_run():
        mlflow.start_run()
        
    try:
        metrics = train_pipeline(r"E:\Project 2\extracted_data\StudentsPerformance.csv")
        print(metrics)
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e
    
    finally:
        mlflow.end_run()