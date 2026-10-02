import mlflow
import os
from dotenv import load_dotenv

load_dotenv()
model_id = os.environ["MODEL_ID"]
mlflow.set_tracking_uri(
        os.environ["DAGSHUB_MLFLOW_TRACKING_URI"]
    )

local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{model_id}",
    dst_path="models/churn_model",
)

print(local_path)