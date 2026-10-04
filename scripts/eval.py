import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

from churn_ml.features.engineering import engineer_features


def load_model():
    load_dotenv()

    tracking_uri = os.getenv("DAGSHUB_MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError(
            "DAGSHUB_MLFLOW_TRACKING_URI is not set."
        )

    mlflow.set_tracking_uri(tracking_uri)

    model_id = os.environ["MODEL_ID"]
    
    return mlflow.sklearn.load_model( f"models:/{model_id}")


def predict_customer(
    model,
    customer: dict,
    threshold: float = 0.08,
):
    df = pd.DataFrame([customer])
    features = engineer_features(df)

    probability = model.predict_proba(features)[0, 1]

    prediction = int(
        probability >= threshold
    )

    return {
        "churn_probability": float(probability),
        "churn_prediction": prediction,
        "churn": "Yes" if prediction == 1 else "No",
        "threshold": threshold,
    }
