import logging
import pandas as pd

from src.model_evaluation import ModelEvaluator, RegressionLinearEvaluationStrategy

def evaluate(model, X_test, y_test) -> dict:
    try:
        evaluator = ModelEvaluator(RegressionLinearEvaluationStrategy())
        evaluated_model = evaluator.evaluate(model, X_test, y_test)
        print(evaluated_model)
        return evaluated_model
    except Exception as e:
        logging.warning(f"Error occur while evaluating model: {e}.")
        raise e