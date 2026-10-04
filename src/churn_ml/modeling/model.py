import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


NUMERIC_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
]


CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "tenure_group",
]

CONTINUOUS_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]



def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

def build_logistic_regression_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "continuous",
                StandardScaler(),
                CONTINUOUS_FEATURES,
            ),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.to_drop_ = [
            column
            for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.drop(columns=self.to_drop_).to_numpy()


def build_model(
        scale_pos_weight: float = 1.0,
        n_jobs: int = 1) -> XGBClassifier:
    return XGBClassifier(
        random_state=42,
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )


def build_pipeline(
        scale_pos_weight: float = 1.0,
        n_jobs: int = 1) -> Pipeline:
    preprocessor = build_preprocessor()
    model = build_model(scale_pos_weight=scale_pos_weight, n_jobs=n_jobs)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ('correlation_filter', CorrelationFilter(threshold=0.99)),
            ("model", model),
        ]
    )

def build_logistic_regression_pipeline(
    C: float = 1.0,
    class_weight: str | None = None,
) -> Pipeline:
    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=2000,
        random_state=42,
    )
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_logistic_regression_preprocessor(),
            ),
            ("correlation_filter", CorrelationFilter(threshold=0.99)),
            ("model", classifier),
        ]
    )