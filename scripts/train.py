from pathlib import Path
import os

import mlflow
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn_ml.data.loader import load_data
from churn_ml.validation.validator import validate_data
from churn_ml.preprocessing.cleaning import clean_data
from churn_ml.features.engineering import engineer_features
from churn_ml.modeling.model import build_pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42
SCALE_POS_WEIGHT = 2.76856

EXPERIMENT_NAME = "telco-churn"


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # 1. Load environment variables
    # -------------------------------------------------------------

    load_dotenv(PROJECT_ROOT / ".env")

    tracking_uri = os.getenv("DAGSHUB_MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise ValueError(
            "DAGSHUB_MLFLOW_TRACKING_URI is not set."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -------------------------------------------------------------
    # 2. Load raw dataset
    # -------------------------------------------------------------

    df = load_data(DATA_PATH)

    # -------------------------------------------------------------
    # 3. Validate
    # -------------------------------------------------------------

    df = validate_data(df)

    # -------------------------------------------------------------
    # 4. Clean
    # -------------------------------------------------------------

    df = clean_data(df)

    # -------------------------------------------------------------
    # 5. Feature engineering
    # -------------------------------------------------------------

    df = engineer_features(df)

    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH),
        name="telco_churn",
    )

    # -------------------------------------------------------------
    # 6. Separate features and target
    # -------------------------------------------------------------

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # -------------------------------------------------------------
    # 7. Train/test split
    # -------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -------------------------------------------------------------
    # 8. Build model pipeline
    # -------------------------------------------------------------

    pipeline = build_pipeline(
        scale_pos_weight=SCALE_POS_WEIGHT
    )

    # -------------------------------------------------------------
    # 9. Start MLflow training run
    # -------------------------------------------------------------

    with mlflow.start_run(run_name="xgboost-baseline-training"):

        # ---------------------------------------------------------
        # Run metadata
        # ---------------------------------------------------------

        mlflow.set_tag("run_type", "training")
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("dataset", DATA_PATH.name)
        mlflow.set_tag("target", "Churn")

        # ---------------------------------------------------------
        # Dataset parameters
        # ---------------------------------------------------------

        mlflow.log_param("dataset_name", DATA_PATH.name)
        mlflow.log_param(
            "dataset_path",
            str(DATA_PATH),
        )
        mlflow.log_param(
            "dataset_rows",
            len(df),
        )
        mlflow.log_param(
            "dataset_features",
            X.shape[1],
        )

        # ---------------------------------------------------------
        # Split parameters
        # ---------------------------------------------------------

        mlflow.log_param(
            "test_size",
            TEST_SIZE,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "train_rows",
            len(X_train),
        )

        mlflow.log_param(
            "test_rows",
            len(X_test),
        )

        # ---------------------------------------------------------
        # Model parameters
        # ---------------------------------------------------------

        mlflow.log_param(
            "model",
            "XGBClassifier",
        )

        mlflow.log_param(
            "scale_pos_weight",
            SCALE_POS_WEIGHT,
        )

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------

        print("Training model...")

        pipeline.fit(
            X_train,
            y_train,
        )

        print("Training complete.")

        # ---------------------------------------------------------
        # Predictions
        # ---------------------------------------------------------

        y_pred = pipeline.predict(X_test)

        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        # ---------------------------------------------------------
        # Evaluation metrics
        # ---------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob,
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob,
        )

        # ---------------------------------------------------------
        # Log metrics
        # ---------------------------------------------------------

        mlflow.log_metric(
            "test_accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "test_precision",
            precision,
        )

        mlflow.log_metric(
            "test_recall",
            recall,
        )

        mlflow.log_metric(
            "test_f1",
            f1,
        )

        mlflow.log_metric(
            "test_roc_auc",
            roc_auc,
        )

        mlflow.log_metric(
            "test_pr_auc",
            pr_auc,
        )

        # ---------------------------------------------------------
        # Log dataset used for training
        # ---------------------------------------------------------

        mlflow.log_artifact(
            str(DATA_PATH),
            artifact_path="dataset",
        )

        # ---------------------------------------------------------
        # Log complete pipeline
        # ---------------------------------------------------------


        mlflow.log_input(dataset, context="training")

        # ---------------------------------------------------------
        # Print results
        # ---------------------------------------------------------

        print("\nTest metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"PR-AUC:    {pr_auc:.4f}")


if __name__ == "__main__":
    main()