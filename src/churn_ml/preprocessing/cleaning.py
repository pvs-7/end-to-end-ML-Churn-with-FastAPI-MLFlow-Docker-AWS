from __future__ import annotations

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco Customer Churn dataset.

    Operations:
    - Remove customerID
    - Convert TotalCharges to numeric
    - Replace missing TotalCharges with 0 for new customers
    - Convert Churn from Yes/No to 1/0
    """

    data = df.copy()

    # Remove identifier
    data = data.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )

    # New customers with tenure == 0 have no TotalCharges yet
    data.loc[
        data["TotalCharges"].isna() & (data["tenure"] == 0),
        "TotalCharges",
    ] = 0.0

    # Make sure there are no unexpected missing TotalCharges values
    if data["TotalCharges"].isna().any():
        raise ValueError(
            "TotalCharges contains missing values that could not be resolved."
        )

    # Convert target to binary
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    if data["Churn"].isna().any():
        raise ValueError(
            "Churn contains unexpected values. Expected only 'Yes' and 'No'."
        )

    data["Churn"] = data["Churn"].astype(int)

    return data