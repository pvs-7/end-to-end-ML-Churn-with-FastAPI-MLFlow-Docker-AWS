# end-to-end-ML-Churn-with-FastAPI-MLFlow-Docker-AWS

## Project Workflow

This repository develops a Telco customer churn classifier and packages it for API inference. The work moves from exploratory analysis to reusable data and model code, then to MLflow tracking and a FastAPI application:

```text
Telco CSV
       ↓
EDA and data-quality investigation (notebook)
       ↓
Validation, cleaning, and feature engineering (Python package)
       ↓
Model training, tuning, and threshold selection
       ↓
MLflow run and model artifact
       ↓
FastAPI prediction endpoint and web form
```

## Data Source

This project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data), published on Kaggle by Blastchar. The raw CSV used by the project is [`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv).

## Notebook and Modularization

[`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) is the exploratory workspace. It loads the raw Telco CSV, inspects columns, data types, missing values, and churn balance, and explores churn rates across customer and service attributes. It also investigates data-cleaning choices such as converting `TotalCharges` and handling blank charges for zero-tenure customers, and compares candidate classifiers with cross-validation metrics. Since churn is imbalanced and the business goal is to identify customers likely to leave, the notebook considers recall, precision, F1, ROC-AUC, and PR-AUC rather than accuracy alone. Later experiments examine feature engineering and the effect of choosing a decision threshold using contact limits and estimated campaign value.

The notebook is for analysis and experimentation; the API does not import or execute it. Stable steps were moved into `src/churn_ml/` so training and inference can reuse the same operations:

| Module | Responsibility |
|---|---|
| `data/loader.py` | Load and check the raw CSV file. |
| `validation/validator.py` | Check required columns, allowed categorical values, and basic numeric constraints. |
| `preprocessing/cleaning.py` | Drop `customerID`, convert `TotalCharges`, resolve valid zero-tenure blanks, and encode the target. |
| `features/engineering.py` | Apply deterministic binary mappings and create `tenure_group`. |
| `modeling/model.py` | Build the `ColumnTransformer`/`OneHotEncoder` preprocessing and XGBoost pipeline. |
| `inference/pipeline.py` | Apply serving-time feature engineering, get churn probability, and apply the selected threshold. |

This separation keeps data preparation, model construction, and prediction logic out of the notebook and gives scripts and the API a common implementation of serving-time features.

## Training and Threshold Selection

The scripts orchestrate the reusable package:

- `scripts/train.py` runs a baseline XGBoost training and test evaluation. It records run parameters, dataset information, metrics, and the source dataset in MLflow.
- `scripts/tune.py` runs an Optuna search with five-fold stratified cross-validation, optimizing recall. It generates out-of-fold probabilities on the training split, selects a decision threshold against the configured contact and campaign-value assumptions, refits the selected pipeline, and evaluates it on the held-out test split. The final pipeline and threshold are logged to MLflow.
- `scripts/eval.py` contains helpers for loading a model by MLflow `MODEL_ID` and scoring a customer record.

The threshold is a separate business decision from the model's probability estimate: a customer is predicted to churn when the probability meets or exceeds the threshold. The tuning script logs the selected threshold as the `selected_threshold` run parameter.

## FastAPI Application

`api/main.py` defines the API and validates prediction requests with the `CustomerRequest` Pydantic model. On application startup, its lifespan handler loads the packaged MLflow model from `models/churn_model` and reads the threshold from `models/metadata.json`. The `/predict` endpoint passes validated customer fields to `churn_ml.inference.pipeline.predict_customer`, which applies feature engineering and returns the churn probability, binary prediction, label, and threshold.

The root route (`GET /`) serves the static dark-mode form in `static/index.html`; the form sends customer attributes to `/predict` and displays those four response values. Customer ID is collected only as a UI reference and is not part of the API request. `GET /health` returns a simple health status for deployment checks.

For local inference, the packaged model directory and metadata file must be present. The API loads these local artifacts at startup, so it does not need to download a model from the tracking server for each request.

## MLflow Model Tracking

Training and tuning use the `telco-churn` MLflow experiment and the `DAGSHUB_MLFLOW_TRACKING_URI` environment variable to select the tracking server. A run captures parameters and tags describing the dataset, model, split, and training configuration, along with evaluation and business metrics. Dataset lineage is recorded with `mlflow.log_input`; the baseline script also logs the raw dataset as an artifact.

The tuning run logs the selected Optuna parameters, out-of-fold threshold-selection metrics, held-out test metrics, the `selected_threshold` parameter, and the fitted scikit-learn pipeline under the `model` artifact name. Because the complete pipeline includes preprocessing as well as the XGBoost estimator, inference can load one MLflow artifact and use the same fitted transformations as training.

The deployment workflow expects an MLflow Registry model identified by `MODEL_ID`, downloads that model, and packages it at `models/churn_model`. It also creates `models/metadata.json` for the threshold consumed by FastAPI. Currently, the workflow writes `0.77` as a fixed threshold; it does not read `selected_threshold` from the tuning run. If the selected business threshold changes, the deployment metadata generation should be updated to use the corresponding tracked value as well.



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

## AWS Deployment & CI/CD

The trained churn prediction model is deployed as a containerized FastAPI application on AWS using **Amazon ECR, ECS/Fargate, and an Application Load Balancer (ALB)**.

The deployment pipeline is automated with **GitHub Actions**, while **MLflow/DagsHub** is used for model tracking and model artifact versioning.

### AWS Architecture

```text
Developer
   │
   │ git push
   ▼
GitHub Repository
   │
   ▼
GitHub Actions
   │
   ├── Continuous Integration
   │    ├── Install dependencies
   │    └── Python syntax validation
   │
   ├── Continuous Delivery
   │    ├── Download model from DagsHub/MLflow
   │    ├── Create model metadata
   │    ├── Build Docker image
   │    └── Push image to Amazon ECR
   │
   └── Continuous Deployment
        ├── Retrieve ECS task definition
        ├── Update container image
        ├── Register new task definition revision
        └── Update ECS service
                    │
                    ▼
             Amazon ECS
              Fargate
                    │
                    ▼
             FastAPI Container
              ┌──────────────┐
              │ XGBoost Model│
              │ FastAPI API  │
              │ /health      │
              │ /predict     │
              └──────┬───────┘
                     │
                     ▼
            Application Load
               Balancer
                     │
                     ▼
                API Client
```

### 1. Docker Container

The FastAPI application and trained model are packaged into a Docker image.

The Docker image contains:

- FastAPI application
- `churn_ml` Python package
- MLflow model artifacts
- Model threshold metadata
- Python dependencies
- Uvicorn application server

The application listens on port `8000`:

```text
0.0.0.0:8000
```

The container also exposes a health endpoint:

```text
GET /health
```

which returns:

```json
{
  "status": "ok"
}
```

### 2. Amazon ECR

**Amazon Elastic Container Registry (ECR)** is used as the Docker image registry.

GitHub Actions builds the image and pushes it to ECR using the Git commit SHA as the image tag.

For example:

```text
<account-id>.dkr.ecr.eu-south-1.amazonaws.com/churn-api:<git-sha>
```

The pipeline also maintains a `latest` tag.

Using the Git SHA as the deployment tag makes each deployment traceable to a specific commit.

### 3. Amazon ECS / Fargate

The Docker image is deployed using **Amazon ECS with AWS Fargate**.

Fargate provides serverless container execution, so no EC2 instances need to be managed.

The ECS service maintains the desired number of running tasks and replaces tasks when a new deployment is released.

The task definition specifies:

- Container image
- CPU and memory
- Container port `8000`
- ECS task execution role
- CloudWatch logging
- Container health check

The application container is configured with:

```text
Port: 8000
Protocol: HTTP
```

### 4. Application Load Balancer

An **Application Load Balancer (ALB)** provides the public HTTP entry point for the API.

The ALB listens on:

```text
HTTP :80
```

and forwards requests to the ECS/Fargate tasks on:

```text
HTTP :8000
```

The target group uses:

```text
Target type: IP
Protocol: HTTP
Port: 8000
```

This allows ECS to automatically register the private IP address of each Fargate task with the target group.

### 5. Health Checks

The ALB continuously checks the FastAPI application using:

```text
GET /health
```

Health check configuration:

```text
Protocol: HTTP
Path: /health
Port: traffic port
Interval: 30 seconds
Timeout: 5 seconds
Healthy threshold: 2
Unhealthy threshold: 2
Success code: 200
```

Traffic is only routed to healthy ECS tasks.

### 6. Security Groups

The deployment uses separate security groups for the ALB and Fargate tasks.

The ALB security group allows inbound HTTP traffic:

```text
Internet
   │
   │ TCP 80
   ▼
ALB Security Group
```

The Fargate security group allows application traffic on port `8000` only from the ALB security group:

```text
ALB Security Group
        │
        │ TCP 8000
        ▼
Fargate Security Group
```

This prevents the application container from being directly exposed to the internet.

### 7. Model Deployment

MLflow/DagsHub remains responsible for **model tracking and versioning**.

During the GitHub Actions build process, the registered model is downloaded from DagsHub/MLflow:

```text
DagsHub / MLflow
       │
       │ download model
       ▼
GitHub Actions
       │
       │ copy into Docker build context
       ▼
Docker Image
       │
       ▼
Amazon ECR
       │
       ▼
ECS/Fargate
```

The Docker image therefore contains the exact model artifact required by the API.

The model is loaded locally by the FastAPI application from:

```text
models/churn_model
```

and the business decision threshold is stored separately in:

```text
models/metadata.json
```

Example:

```json
{
  "threshold": 0.77
}
```

This separates **model versioning** from **application deployment** while ensuring that each deployed container contains a reproducible model artifact.

### 8. GitHub Actions CI/CD

The GitHub Actions workflow is triggered when changes are pushed to the `main` branch.

README-only changes are ignored.

The workflow consists of three stages:

#### Continuous Integration

```text
Install dependencies
       ↓
Python syntax validation
```

#### Continuous Delivery

```text
Download model from DagsHub
       ↓
Create model metadata
       ↓
Build Docker image
       ↓
Push image to ECR
```

#### Continuous Deployment

```text
Retrieve current ECS task definition
       ↓
Replace container image with new Git SHA
       ↓
Register new ECS task definition revision
       ↓
Update ECS service
       ↓
Wait for service stability
```

This means a normal Git push is sufficient to deploy a new version:

```bash
git add .
git commit -m "Update API"
git push origin main
```

The deployment then proceeds automatically:

```text
GitHub
  ↓
GitHub Actions
  ↓
ECR
  ↓
ECS
  ↓
Fargate
  ↓
ALB
  ↓
FastAPI
```

### 9. AWS IAM

Two different IAM roles are used for different responsibilities.

#### ECS Task Execution Role

The ECS task execution role allows Fargate to perform AWS operations required to start the container, such as pulling the image from ECR and sending logs to CloudWatch.

The role used by the ECS tasks is:

```text
ecsTaskExecutionRole
```

#### GitHub Actions AWS Identity

GitHub Actions uses an AWS IAM identity with permissions required to:

- Push Docker images to ECR
- Read ECS task definitions
- Register new ECS task definition revisions
- Update the ECS service
- Describe ECS services
- Pass the ECS task execution role

This separates deployment permissions from the permissions used by the running application.

### 10. Logging

Container logs are sent to **Amazon CloudWatch Logs**.

The ECS task uses the `awslogs` log driver and writes application logs to:

```text
/ecs/churn-api
```

This makes container startup errors, FastAPI logs, and deployment issues available through the AWS console.

### Deployment Result

The final deployment provides:

- Containerized ML inference
- Serverless container execution with Fargate
- Public API access through an ALB
- Automated health checks
- CloudWatch logging
- ECR image versioning
- MLflow/DagsHub model tracking
- Automated GitHub Actions CI/CD
- Reproducible deployments using Git commit SHA image tags

The resulting workflow is:

```text
Git Commit
    ↓
GitHub Actions
    ↓
Model Download
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
ECS Task Definition
    ↓
ECS/Fargate
    ↓
ALB Health Check
    ↓
Healthy FastAPI Service
```