from pathlib import Path
import os

import mlflow
import optuna
from dotenv import load_dotenv

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    train_test_split,
)

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

N_TRIALS = 50

EXPERIMENT_NAME = "telco-churn"


# Business assumptions
CONTACT_COST = 10
SAVE_RATE = 0.25
RETAINED_CUSTOMER_VALUE = 200
MAX_CONTACTS = 1000


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def prepare_data():

    df = load_data(DATA_PATH)

    df = validate_data(df)

    df = clean_data(df)

    df = engineer_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return df, X, y


# ---------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------

def create_objective(X_train, y_train):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                800,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                8,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0,
                5,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                5,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                5,
                log=True,
            ),
        }

        pipeline = build_pipeline(
            scale_pos_weight=SCALE_POS_WEIGHT
        )

        # Apply Optuna parameters to the XGBoost step.
        pipeline.set_params(
            **{
                f"model__{key}": value
                for key, value in params.items()
            }
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="recall",
            n_jobs=-1,
        )

        return scores.mean()

    return objective


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def select_threshold(
    y_true,
    probabilities,
    max_contacts=MAX_CONTACTS,
    contact_cost=CONTACT_COST,
    save_rate=SAVE_RATE,
    retained_customer_value=RETAINED_CUSTOMER_VALUE,
):
    """
    Select threshold using only training OOF predictions.

    Objective:
        maximize expected net business value

    Constraint:
        contacts <= max_contacts
    """

    thresholds = sorted(
        set(probabilities)
    )

    best_result = None

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        contacts = tp + fp

        if contacts > max_contacts:
            continue

        expected_saved = tp * save_rate

        campaign_cost = (
            contacts * contact_cost
        )

        retained_value = (
            expected_saved
            * retained_customer_value
        )

        net_value = (
            retained_value
            - campaign_cost
        )

        if (
            best_result is None
            or net_value > best_result["net_value"]
        ):
            best_result = {
                "threshold": threshold,
                "contacts": contacts,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "expected_saved": expected_saved,
                "campaign_cost": campaign_cost,
                "retained_value": retained_value,
                "net_value": net_value,
            }

    if best_result is None:
        raise ValueError(
            "No threshold satisfies the contact constraint."
        )

    return best_result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # 1. Environment
    # -------------------------------------------------------------

    load_dotenv(PROJECT_ROOT / ".env")

    tracking_uri = os.getenv(
        "DAGSHUB_MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise ValueError(
            "DAGSHUB_MLFLOW_TRACKING_URI is not set."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -------------------------------------------------------------
    # 2. Prepare data
    # -------------------------------------------------------------

    df, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -------------------------------------------------------------
    # 3. Create dataset lineage
    # -------------------------------------------------------------

    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH),
        name="telco_churn_processed",
    )

    # -------------------------------------------------------------
    # 4. MLflow run
    # -------------------------------------------------------------

    with mlflow.start_run(
        run_name="xgboost-optuna-tuning"
    ):

        mlflow.set_tag(
            "run_type",
            "hyperparameter_tuning",
        )

        mlflow.set_tag(
            "model_type",
            "XGBoost",
        )

        mlflow.set_tag(
            "target",
            "Churn",
        )

        mlflow.log_input(
            dataset,
            context="training",
        )

        # ---------------------------------------------------------
        # 5. Log configuration
        # ---------------------------------------------------------

        mlflow.log_params({
            "dataset_name": DATA_PATH.name,
            "dataset_rows": len(df),
            "input_features": X.shape[1],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "scale_pos_weight": SCALE_POS_WEIGHT,
            "optuna_n_trials": N_TRIALS,
            "cv_folds": 5,
            "optimization_metric": "recall",
            "max_contacts": MAX_CONTACTS,
            "contact_cost": CONTACT_COST,
            "save_rate": SAVE_RATE,
            "retained_customer_value": RETAINED_CUSTOMER_VALUE,
        })

        # ---------------------------------------------------------
        # 6. Optuna study
        # ---------------------------------------------------------

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE
            ),
        )

        objective = create_objective(
            X_train,
            y_train,
        )

        print("Starting Optuna tuning...")

        study.optimize(
            objective,
            n_trials=N_TRIALS,
        )

        print("Optuna tuning complete.")

        # ---------------------------------------------------------
        # 7. Best parameters
        # ---------------------------------------------------------

        best_params = study.best_params

        print("\nBest parameters:")
        for name, value in best_params.items():
            print(f"{name}: {value}")

        print(
            f"\nBest CV recall: "
            f"{study.best_value:.4f}"
        )

        mlflow.log_metric(
            "best_cv_recall",
            study.best_value,
        )

        mlflow.log_param(
            "optuna_best_trial",
            study.best_trial.number,
        )

        mlflow.log_params({
            f"best_{key}": value
            for key, value in best_params.items()
        })

        # ---------------------------------------------------------
        # 8. Build best pipeline
        # ---------------------------------------------------------

        best_pipeline = build_pipeline(
            scale_pos_weight=SCALE_POS_WEIGHT
        )

        best_pipeline.set_params(
            **{
                f"model__{key}": value
                for key, value in best_params.items()
            }
        )

        # ---------------------------------------------------------
        # 9. Generate OOF probabilities
        # ---------------------------------------------------------

        print("\nGenerating OOF probabilities...")

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        oof_probabilities = cross_val_predict(
            best_pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        # ---------------------------------------------------------
        # 10. Select business threshold
        # ---------------------------------------------------------

        threshold_result = select_threshold(
            y_train,
            oof_probabilities,
        )

        threshold = threshold_result["threshold"]

        print("\nSelected threshold:")
        print(f"Threshold: {threshold:.3f}")
        print(
            f"Contacts: {threshold_result['contacts']}"
        )
        print(
            f"Recall: {threshold_result['recall']:.4f}"
        )
        print(
            f"Precision: "
            f"{threshold_result['precision']:.4f}"
        )
        print(
            f"Expected net value: "
            f"€{threshold_result['net_value']:,.2f}"
        )

        # ---------------------------------------------------------
        # 11. Log threshold metrics
        # ---------------------------------------------------------

        mlflow.log_param(
            "selected_threshold",
            threshold,
        )

        mlflow.log_metrics({
            "oof_recall": threshold_result["recall"],
            "oof_precision": threshold_result["precision"],
            "oof_contacts": threshold_result["contacts"],
            "oof_tp": threshold_result["tp"],
            "oof_fp": threshold_result["fp"],
            "oof_tn": threshold_result["tn"],
            "oof_fn": threshold_result["fn"],
            "oof_expected_saved": threshold_result[
                "expected_saved"
            ],
            "oof_campaign_cost": threshold_result[
                "campaign_cost"
            ],
            "oof_retained_value": threshold_result[
                "retained_value"
            ],
            "oof_net_value": threshold_result[
                "net_value"
            ],
        })

        # ---------------------------------------------------------
        # 12. Refit on ALL training data
        # ---------------------------------------------------------

        print("\nTraining final model...")

        best_pipeline.fit(
            X_train,
            y_train,
        )

        # ---------------------------------------------------------
        # 13. Final test predictions
        # ---------------------------------------------------------

        test_probabilities = (
            best_pipeline.predict_proba(
                X_test
            )[:, 1]
        )

        test_predictions = (
            test_probabilities >= threshold
        ).astype(int)

        # ---------------------------------------------------------
        # 14. Test metrics
        # ---------------------------------------------------------

        test_accuracy = accuracy_score(
            y_test,
            test_predictions,
        )

        test_precision = precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )

        test_recall = recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )

        test_f1 = f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )

        test_roc_auc = roc_auc_score(
            y_test,
            test_probabilities,
        )

        test_pr_auc = average_precision_score(
            y_test,
            test_probabilities,
        )

        # ---------------------------------------------------------
        # 15. Log final test metrics
        # ---------------------------------------------------------

        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_roc_auc": test_roc_auc,
            "test_pr_auc": test_pr_auc,
        })

        # ---------------------------------------------------------
        # 16. Final business metrics
        # ---------------------------------------------------------

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1],
        ).ravel()

        contacts = tp + fp

        expected_saved = (
            tp * SAVE_RATE
        )

        campaign_cost = (
            contacts * CONTACT_COST
        )

        retained_value = (
            expected_saved
            * RETAINED_CUSTOMER_VALUE
        )

        net_value = (
            retained_value
            - campaign_cost
        )

        mlflow.log_metrics({
            "test_contacts": contacts,
            "test_tp": tp,
            "test_fp": fp,
            "test_tn": tn,
            "test_fn": fn,
            "test_expected_saved": expected_saved,
            "test_campaign_cost": campaign_cost,
            "test_retained_value": retained_value,
            "test_net_value": net_value,
        })

        # ---------------------------------------------------------
        # 17. Log final model
        # ---------------------------------------------------------

        mlflow.sklearn.log_model(
            best_pipeline,
            name="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        # ---------------------------------------------------------
        # 18. Print results
        # ---------------------------------------------------------

        print("\nFinal test results:")
        print(f"Accuracy:  {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1:        {test_f1:.4f}")
        print(f"ROC-AUC:   {test_roc_auc:.4f}")
        print(f"PR-AUC:    {test_pr_auc:.4f}")

        print("\nFinal business results:")
        print(f"Threshold: {threshold:.3f}")
        print(f"Contacts:  {contacts}")
        print(f"TP:        {tp}")
        print(f"FP:        {fp}")
        print(f"FN:        {fn}")
        print(
            f"Net value: €{net_value:,.2f}"
        )


if __name__ == "__main__":
    main()