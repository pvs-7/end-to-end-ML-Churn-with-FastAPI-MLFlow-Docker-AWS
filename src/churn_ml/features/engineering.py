from __future__ import annotations

import pandas as pd


BINARY_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic feature engineering to the cleaned dataset.
    """

    data = df.copy()

    # Convert binary categorical features to 0/1
    binary_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }

    for column, mapping in binary_mappings.items():
        data[column] = data[column].map(mapping)

        if data[column].isna().any():
            raise ValueError(
                f"Unexpected values found in binary column: {column}"
            )

        data[column] = data[column].astype("int8")

    # Create tenure groups
    bins = [-1, 6, 12, 24, 48, 72]
    labels = [
        "0-6",
        "7-12",
        "13-24",
        "25-48",
        "49-72",
    ]

    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=bins,
        labels=labels,
    )

    data["tenure_group"] = data["tenure_group"].astype(str)

    return data