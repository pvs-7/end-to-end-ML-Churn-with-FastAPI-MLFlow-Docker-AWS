from pathlib import Path
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

from churn_ml.data.loader import load_data
from churn_ml.features.engineering import engineer_features
from churn_ml.modeling.model import build_logistic_regression_pipeline
from churn_ml.preprocessing.cleaning import clean_data
from churn_ml.validation.validator import validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_TRIALS = 150
INNER_SPLITS = 5
EXPERIMENT_NAME = "Telco Churn - Logistic Regression"

CONTACT_COST = 120.0
SAVE_RATE = 0.45
THRESHOLDS = np.linspace(0.05, 0.95, 91)


def prepare_data():
    df = load_data(DATA_PATH)
    df = validate_data(df)
    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return df, X, y


def create_objective(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced"],
            ),
        }
        pipeline = build_logistic_regression_pipeline(**params)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
        return float(scores.mean())

    return objective


def campaign_metrics(y_true, y_pred, monthly_charges):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ltv = np.asarray(monthly_charges, dtype=float) * 12

    is_tp = (y_true == 1) & (y_pred == 1)
    is_fp = (y_true == 0) & (y_pred == 1)
    is_fn = (y_true == 1) & (y_pred == 0)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    contacts = tp + fp

    true_positive_gain = float(
        np.sum(ltv[is_tp] * SAVE_RATE)
        - np.sum(is_tp) * CONTACT_COST
    )
    false_positive_loss = float(fp * CONTACT_COST)
    false_negative_loss = float(np.sum(ltv[is_fn]))

    return {
        "contacts": int(contacts),
        "contact_rate": float(contacts / len(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "true_positive_gain": true_positive_gain,
        "false_positive_loss": false_positive_loss,
        "false_negative_loss": false_negative_loss,
        "campaign_cost": float(contacts * CONTACT_COST),
        "total_tp_ltv": float(np.sum(ltv[is_tp])),
        "expected_retained_value": float(
            np.sum(ltv[is_tp]) * SAVE_RATE
        ),
        "total_fn_ltv": float(np.sum(ltv[is_fn])),
        "net_value": float(
            true_positive_gain
            - false_positive_loss
            - false_negative_loss
        ),
    }


def select_threshold(
    y_true,
    probabilities,
    monthly_charges,
):
    best_result = None
    best_threshold = None

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        result = campaign_metrics(
            y_true,
            predictions,
            monthly_charges,
        )
        if best_result is None or result["net_value"] > best_result["net_value"]:
            best_result = result
            best_threshold = float(threshold)

    if best_result is None or best_threshold is None:
        raise RuntimeError("Threshold search did not produce a result.")

    best_result["threshold"] = best_threshold
    best_result["recall"] = float(
        recall_score(y_true, (probabilities >= best_threshold), zero_division=0)
    )
    best_result["precision"] = float(
        precision_score(y_true, (probabilities >= best_threshold), zero_division=0)
    )
    return best_result


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("DAGSHUB_MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("DAGSHUB_MLFLOW_TRACKING_URI is not set.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH),
        name="telco_churn_processed",
    )

    with mlflow.start_run(run_name="LogisticRegression_nested_cv_final"):
        mlflow.set_tags({
            "model": "Logistic Regression",
            "stage": "final_evaluation",
            "preprocessing": (
                "fold-fitted sklearn pipeline; scaled continuous features"
            ),
            "threshold_source": "training_oof_predictions",
            "threshold_optimization": "business_net_value",
            "business_value_method": "customer_specific_ltv",
        })
        mlflow.log_input(dataset, context="training")
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "dataset_name": DATA_PATH.name,
            "dataset_rows": len(df),
            "input_features": X.shape[1],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "optuna_trials": N_TRIALS,
            "inner_cv_splits": INNER_SPLITS,
            "optimization_metric": "average_precision",
            "contact_cost": CONTACT_COST,
            "save_rate": SAVE_RATE,
            "ltv_definition": "MonthlyCharges * 12",
        })

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            create_objective(X_train, y_train),
            n_trials=N_TRIALS,
        )

        best_params = study.best_params
        best_pipeline = build_logistic_regression_pipeline(**best_params)
        mlflow.log_params({
            f"best_{key}": value
            for key, value in best_params.items()
        })
        mlflow.log_param("optuna_best_trial", study.best_trial.number)
        mlflow.log_metric("best_training_cv_pr_auc", study.best_value)

        threshold_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + 100,
        )
        oof_probabilities = cross_val_predict(
            best_pipeline,
            X_train,
            y_train,
            cv=threshold_cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        oof_pr_auc = average_precision_score(y_train, oof_probabilities)
        threshold_result = select_threshold(
            y_train.to_numpy(),
            oof_probabilities,
            X_train["MonthlyCharges"].to_numpy(dtype=float),
        )
        threshold = threshold_result["threshold"]

        mlflow.log_params({
            "decision_threshold": threshold,
            "selected_threshold": threshold,
            "threshold_grid_min": float(THRESHOLDS.min()),
            "threshold_grid_max": float(THRESHOLDS.max()),
            "threshold_grid_steps": len(THRESHOLDS),
        })
        mlflow.log_metrics({
            "oof_pr_auc": oof_pr_auc,
            "oof_recall": threshold_result["recall"],
            "oof_precision": threshold_result["precision"],
            "oof_contacts": threshold_result["contacts"],
            "oof_tp": threshold_result["tp"],
            "oof_fp": threshold_result["fp"],
            "oof_tn": threshold_result["tn"],
            "oof_fn": threshold_result["fn"],
            "oof_net_value": threshold_result["net_value"],
        })

        best_pipeline.fit(X_train, y_train)
        test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= threshold).astype(int)

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "test_precision": precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
            "test_recall": recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
            "test_f1": f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
            "test_roc_auc": roc_auc_score(y_test, test_probabilities),
            "test_pr_auc": average_precision_score(
                y_test,
                test_probabilities,
            ),
        }
        test_business = campaign_metrics(
            y_test.to_numpy(),
            test_predictions,
            X_test["MonthlyCharges"].to_numpy(dtype=float),
        )
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in test_business.items()
            if key != "threshold"
        })

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            test_predictions,
            labels=[0, 1],
            ax=ax,
        )
        ax.set_title(
            f"Logistic Regression test confusion matrix "
            f"(threshold={threshold:.2f})"
        )
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix_logistic_regression.png")
        plt.close(fig)

        mlflow.sklearn.log_model(
            best_pipeline,
            name="model",
            skops_trusted_types=[
                "churn_ml.modeling.model.CorrelationFilter",
            ],
        )

        print(f"Best CV PR-AUC: {study.best_value:.3f}")
        print(f"OOF PR-AUC: {oof_pr_auc:.3f}")
        print(f"Threshold: {threshold:.3f}")
        print(f"Test PR-AUC: {test_metrics['test_pr_auc']:.3f}")
        print(f"Test recall: {test_metrics['test_recall']:.3%}")
        print(f"Test precision: {test_metrics['test_precision']:.3%}")
        print(f"Contacts: {test_business['contacts']}")
        print(f"Estimated net value: €{test_business['net_value']:,.2f}")


if __name__ == "__main__":
    main()
