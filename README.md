# end-to-end-ML-Churn-with-FastAPI-MLFlow-Docker-AWS

EDA
 ↓
Train/test split
 ↓
Preprocessing fitted on training data
 ↓
Baseline model
 ↓
Evaluation
 ↓
Feature engineering / selection
 ↓
Tuning
 ↓
Final test evaluation

Logistic Regression
Random Forest
Gradient Boosting
XGBoost / LightGBM (if available)

## MLflow Model Inference

The final model is logged to MLflow as a complete scikit-learn pipeline containing both preprocessing and the XGBoost classifier.

### Current approach: `mlflow.sklearn`

The current implementation loads the model using MLflow's sklearn flavor:

```python
model = mlflow.sklearn.load_model(model_uri)
```

The application then calls `predict_proba()` to obtain the churn probability:

```python
probability = float(
    model.predict_proba(features)[0, 1]
)
```

The probability is then compared against the business-selected threshold:

```python
prediction = int(probability >= threshold)
```

This approach is useful for this project because the application needs the probability of the positive class (`Churn = 1`) rather than only the predicted class.

The inference flow is:

```text
Customer input
      ↓
Feature engineering
      ↓
MLflow sklearn pipeline
      ↓
predict_proba()
      ↓
P(Churn = 1)
      ↓
Business threshold = 0.77
      ↓
Churn prediction
```

### Alternative approach: MLflow PyFunc

MLflow also provides a generic PyFunc interface. By default, an sklearn model logged through MLflow uses the underlying model's `predict()` method:

```yaml
predict_fn: predict
```

Therefore:

```python
model = mlflow.pyfunc.load_model(model_uri)

prediction = model.predict(features)
```

returns the model's class predictions (`0` or `1`), not the churn probabilities required by this application.

To make the PyFunc interface return probabilities instead, the model can be logged with:

```python
mlflow.sklearn.log_model(
    best_pipeline,
    name="model",
    pyfunc_predict_fn="predict_proba",
    skops_trusted_types=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
    ],
)
```

The resulting MLflow `MLmodel` configuration uses:

```yaml
predict_fn: predict_proba
```

The application can then use:

```python
model = mlflow.pyfunc.load_model(model_uri)

probabilities = model.predict(features)
probability = float(probabilities[0, 1])
```

In this case, PyFunc's generic `predict()` method delegates to the underlying pipeline's `predict_proba()` method.

### Comparison

| Approach | Model loading | Prediction call | Output |
|---|---|---|---|
| Current | `mlflow.sklearn.load_model()` | `predict_proba()` | Class probabilities |
| PyFunc | `mlflow.pyfunc.load_model()` | `predict()` | Class probabilities, if logged with `pyfunc_predict_fn="predict_proba"` |
| PyFunc default | `mlflow.pyfunc.load_model()` | `predict()` | Class predictions |

The important distinction is that **PyFunc does not automatically mean probability predictions**. Its `predict()` behavior depends on the `predict_fn` configured in the saved MLflow model artifact.

For this project, both approaches can provide the required churn probability. The current sklearn approach is more explicit, while the PyFunc approach provides a standardized `model.predict()` interface.