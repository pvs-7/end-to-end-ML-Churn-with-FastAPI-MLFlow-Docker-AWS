# Telco Customer Churn Prediction

This project develops a Telco customer churn model, tracks it with MLflow, and
serves predictions through a FastAPI application. The repository contains an
exploratory notebook, reusable data and modeling code, training scripts, and
AWS deployment automation.

## Project workflow

```text
Telco customer data
        ↓
Exploration and data-quality checks
        ↓
Reusable cleaning and feature engineering
        ↓
Cross-validated model and threshold selection
        ↓
MLflow model and run tracking
        ↓
FastAPI inference
        ↓
Docker and AWS deployment
```

## Data and project structure

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)
published by Blastchar on Kaggle. The expected raw file is
[`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`](data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv).

[`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) contains data exploration,
feature experiments, model comparisons, and the nested cross-validation
analysis. Reusable application and training logic lives in `src/churn_ml/`:

| Module | Purpose |
|---|---|
| `data/loader.py` | Load the raw dataset. |
| `validation/validator.py` | Check required fields, categorical values, and basic numeric constraints. |
| `preprocessing/cleaning.py` | Drop identifiers, convert `TotalCharges`, handle valid blank charges, and encode the target. |
| `features/engineering.py` | Apply deterministic feature mappings and create `tenure_group`. |
| `modeling/model.py` | Build model preprocessing and estimator pipelines. |
| `inference/pipeline.py` | Prepare a customer record, obtain churn probability, and apply the decision threshold. |

The API does not execute the notebook. It uses the shared package so its
feature preparation is consistent with the packaged model.

## Model evaluation and selection

### From the earlier approach to nested cross-validation

The earlier workflow tuned a model using cross-validation, generated
out-of-fold (OOF) probabilities on the training partition, and selected a
decision threshold that maximized estimated campaign value. This is useful
for developing a model and choosing a threshold, but a model or threshold
comparison made on those same OOF labels can be optimistic: the threshold was
chosen to perform well on those observations.

The nested cross-validation approach evaluates the complete selection process
on outer folds:

1. Split the development data into outer training and validation folds.
2. Within each outer training fold, use inner stratified cross-validation and
   Optuna to tune model hyperparameters for Average Precision (PR-AUC).
3. Generate OOF probabilities within the outer training fold and choose the
   campaign threshold using the business-value objective.
4. Fit the tuned model on the outer training fold and score its untouched
   outer validation fold using the selected threshold.
5. Summarize the metrics and campaign value across outer folds.

This provides a more realistic comparison of the whole model-and-threshold
selection procedure, rather than just comparing models at a default threshold
or reporting the value used to choose the threshold. The notebook's nested
analysis uses five outer folds and five inner folds.

For practical runtime, the notebook's threshold OOF predictions use
hyperparameters selected on the full outer-training fold. Each OOF observation
is left out of its corresponding model fit, but its label may have influenced
the hyperparameter choice. The untouched outer validation fold evaluates the
resulting procedure. Fully independent threshold-calibration predictions
would require another hyperparameter-tuning loop inside each threshold fold
and considerably more computation.

### Nested cross-validation results

The table reports means and standard deviations across the outer folds from
the notebook. Campaign value is an estimate under the assumptions below, not
measured campaign profit.

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Contacts | Estimated net value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.094 ± 0.023 | 0.385 ± 0.024 | 0.948 ± 0.031 | 0.547 ± 0.018 | 0.846 ± 0.011 | 0.663 ± 0.025 | 740 ± 66 | €17,157 ± €3,786 |
| XGBoost | 0.266 ± 0.038 | 0.414 ± 0.017 | 0.924 ± 0.026 | 0.572 ± 0.011 | 0.850 ± 0.013 | 0.671 ± 0.023 | 668 ± 44 | €17,030 ± €5,410 |
| Logistic Regression | 0.104 ± 0.021 | 0.401 ± 0.016 | 0.940 ± 0.020 | 0.562 ± 0.013 | 0.848 ± 0.012 | 0.663 ± 0.015 | 702 ± 43 | €17,030 ± €3,496 |
| Gradient Boosting | 0.108 ± 0.013 | 0.410 ± 0.011 | 0.929 ± 0.022 | 0.569 ± 0.008 | 0.849 ± 0.013 | 0.668 ± 0.023 | 678 ± 34 | €16,994 ± €5,435 |

The campaign-value calculation in this comparison assumes:

- Customer lifetime value is approximated as `MonthlyCharges * 12`.
- A successful retention intervention saves 45% of that estimated value.
- Each contacted customer costs €120.
- False negatives incur the estimated value of the customer who churns.

These are modeling assumptions; replace them with validated campaign costs,
retention effectiveness, and customer-value estimates when available.

### Why Logistic Regression is the deployment candidate

The outer-fold results are close, and no model dominates every metric.
Random Forest has the highest mean estimated campaign value in this
comparison, while XGBoost has the highest mean PR-AUC. Logistic Regression
nevertheless has competitive campaign value, high recall, and PR-AUC close to
the top result, with less variation in campaign value than XGBoost or
Gradient Boosting in these folds.

Logistic Regression is used for the final modular training workflow as a
deliberate balance of predictive performance, fold-to-fold consistency,
simplicity, and ease of explaining and maintaining the model. This is a
practical model choice, not a claim that it has the best score on every
metric. The selected decision threshold is a separate business decision from
the model itself and is chosen using training-only OOF probabilities.

### XGBoost as an alternative

XGBoost remains a reasonable alternative when its higher PR-AUC or higher
precision at the chosen operating point is more important to the campaign.
The nested comparison estimates a slightly higher PR-AUC for XGBoost, but
also shows greater variability in its campaign-value estimate. It can be
selected instead by repeating the same nested comparison with business
assumptions and operating priorities agreed in advance.

[`scripts/train.py`](scripts/train.py) provides an XGBoost baseline, while
[`scripts/tune.py`](scripts/tune.py) implements the current Logistic
Regression tuning and threshold workflow. The inference service loads a
complete MLflow scikit-learn pipeline, so either estimator can be served when
the appropriate pipeline is logged and deployed. The deployment environment
must include the dependencies required by the selected estimator.

## Training, threshold selection, and MLflow

[`scripts/tune.py`](scripts/tune.py) prepares the data, creates a stratified
training/test split, and runs an Optuna search for Logistic Regression. The
search tunes regularization strength (`C`) and class weighting using
five-fold cross-validation and Average Precision. It then generates
training-only OOF probabilities with stratified folds and selects the
threshold that maximizes the configured campaign net-value estimate.

After selection, the script fits the pipeline on the training partition and
evaluates it using the chosen threshold. The run records parameters,
cross-validation and evaluation metrics, dataset lineage, the threshold, and
the fitted preprocessing-plus-model pipeline in MLflow. The final threshold
is also logged as `selected_threshold`.

The binary decision rule is:

```text
predict churn when churn_probability >= selected_threshold
```

The probability threshold controls how many customers are contacted and the
precision/recall trade-off; it is not a parameter learned by the classifier.
The nested-CV comparison is for choosing the model and procedure. The modular
training script is the practical final-fit workflow.

[`scripts/eval.py`](scripts/eval.py) provides helpers for loading a logged
model and scoring a customer record.

MLflow tracking is configured through `DAGSHUB_MLFLOW_TRACKING_URI` and the
MLflow credentials in the environment.
The deployment workflow uses `MODEL_ID` to identify the MLflow model artifact.

## FastAPI inference

[`api/main.py`](api/main.py) defines the FastAPI service. At startup, it loads
the packaged model from `models/churn_model` and reads the threshold from
`models/metadata.json`. The prediction endpoint validates customer input,
applies shared feature engineering, and returns the churn probability,
binary prediction, label, and threshold.

- `GET /` serves the web form.
- `GET /health` is used for health checks.
- `POST /predict` returns a customer churn prediction.
- `/docs` provides the interactive API documentation.

For local execution, install the project dependencies with `uv sync`, then
start the service from the repository root with
`uv run uvicorn api.main:app --reload`. Open `http://127.0.0.1:8000/docs` to
try the API. Local inference requires the model directory and metadata file
to be present.

## AWS deployment and CI/CD

The API is deployed as a Docker container on **Amazon ECS with AWS Fargate**.
**Amazon ECR** stores versioned images, and an **Application Load Balancer
(ALB)** provides the HTTP entry point. **GitHub Actions** automates
integration checks, image delivery, and ECS deployment. MLflow/DagsHub stores
and tracks model artifacts.

### Architecture

```text
GitHub push to main
        ↓
GitHub Actions: install dependencies and check Python syntax
        ↓
Download model artifact from DagsHub/MLflow using MODEL_ID
        ↓
Create model threshold metadata and build Docker image
        ↓
Push image tagged with the Git commit SHA to Amazon ECR
        ↓
Update ECS task definition and ECS service
        ↓
Amazon ECS / Fargate runs the FastAPI container
        ↓
Application Load Balancer routes requests to healthy tasks
```

README-only changes are excluded from the deployment workflow. Each image is
tagged with its Git commit SHA for traceability; the workflow also maintains
a `latest` tag.

### Container, networking, and health checks

The image contains the FastAPI application, the `churn_ml` package, the
packaged MLflow model, threshold metadata, Python dependencies, and Uvicorn.
The service listens on container port `8000`. The `GET /health` endpoint
returns a successful status for the load balancer's health checks.

The ALB listens on HTTP port `80` and forwards requests to Fargate tasks on
port `8000`. The target group uses IP targets, as required for Fargate
networking. The ALB security group accepts public HTTP traffic; the task
security group permits port `8000` only from the ALB security group, keeping
the application tasks from being directly exposed to the internet.

The target group's health check uses `GET /health` over HTTP on the traffic
port. Traffic is routed only to healthy tasks. ECS/Fargate maintains the
desired number of tasks and replaces them as part of service deployments.

### Model packaging

During image creation, GitHub Actions downloads the selected model artifact
from DagsHub/MLflow and copies it into the Docker build context at
`models/churn_model`. The FastAPI service loads that local artifact at
startup; it does not download a model for each prediction request.
`models/metadata.json` contains the decision threshold consumed by inference.

The current deployment workflow writes a threshold of `0.08` when creating
that metadata. If the chosen threshold changes, update the deployment
metadata step to keep the packaged threshold aligned with the MLflow run.

### GitHub Actions workflow

The deployment workflow in
[`deploy.yml`](.github/workflows/deploy.yml) runs on pushes to `main`:

1. **Continuous integration:** install dependencies and validate Python
   syntax.
2. **Continuous delivery:** download and verify the MLflow model, create
   metadata, build the Docker image, and push it to ECR.
3. **Continuous deployment:** render a new ECS task definition using the
   commit-tagged image, update the ECS service, and wait for service stability.

Configure these GitHub Actions secrets for the workflow:

- `AWS_REGION`
- `ECR_REPOSITORY_NAME`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `DAGSHUB_MLFLOW_TRACKING_URI`
- `DAGSHUB_MLFLOW_TRACKING_USERNAME`
- `DAGSHUB_MLFLOW_TRACKING_PASSWORD`
- `MODEL_ID`
- `ECS_TASK_DEFINITION`
- `ECS_SERVICE_NAME`
- `ECS_CLUSTER_NAME`

The GitHub Actions AWS identity needs permission to push images to ECR, read
and register ECS task definitions, update and describe the ECS service, and
pass the ECS task execution role. The ECS task execution role allows Fargate
to pull the image and send logs to CloudWatch. Application container logs
are written to the `/ecs/churn-api` log group through the `awslogs` driver.

## License and attribution

The dataset is provided by Blastchar on Kaggle. Follow the dataset
publisher's terms when using or redistributing it.
