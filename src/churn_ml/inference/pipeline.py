
import pandas as pd
from churn_ml.features.engineering import engineer_features

def predict_customer(
    customer_data: dict,
    model,
    threshold: float = 0.08,
) -> dict:
    customer_df = pd.DataFrame([customer_data])

    features = engineer_features(customer_df)

    probability = float(model.predict_proba(features)[0, 1])

    prediction = int(probability >= threshold)

    return {
        "churn_probability": probability,
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction else "No",
        "threshold": threshold,
    }
