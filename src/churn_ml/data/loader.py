from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.

    Parameters
    ----------
    path : str or Path
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got {path.suffix}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df