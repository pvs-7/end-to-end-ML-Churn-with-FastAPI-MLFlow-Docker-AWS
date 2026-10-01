from __future__ import annotations

import pandas as pd


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw Telco dataset."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Check categorical values
    expected_values = {
        "gender": {"Male", "Female"},
        "Partner": {"Yes", "No"},
        "Dependents": {"Yes", "No"},
        "PhoneService": {"Yes", "No"},
        "MultipleLines": {"Yes", "No", "No phone service"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "OnlineSecurity": {"Yes", "No", "No internet service"},
        "OnlineBackup": {"Yes", "No", "No internet service"},
        "DeviceProtection": {"Yes", "No", "No internet service"},
        "TechSupport": {"Yes", "No", "No internet service"},
        "StreamingTV": {"Yes", "No", "No internet service"},
        "StreamingMovies": {"Yes", "No", "No internet service"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "PaperlessBilling": {"Yes", "No"},
        "PaymentMethod": {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        },
        "Churn": {"Yes", "No"},
    }

    for column, allowed_values in expected_values.items():
        actual_values = set(df[column].dropna().unique())
        unexpected_values = actual_values - allowed_values

        if unexpected_values:
            raise ValueError(
                f"Unexpected values in '{column}': "
                f"{unexpected_values}"
            )

    # Check numeric ranges
    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError(
            "SeniorCitizen must contain only 0 or 1."
        )

    if (df["tenure"] < 0).any():
        raise ValueError("tenure cannot be negative.")

    if not (df["MonthlyCharges"] >= 0).all():
        raise ValueError(
            "MonthlyCharges cannot be negative."
        )

    return df