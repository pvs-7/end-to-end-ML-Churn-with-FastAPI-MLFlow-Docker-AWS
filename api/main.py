from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Literal
import mlflow
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from churn_ml.inference.pipeline import predict_customer

# mlflow model artifacts
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "churn_model"
)

METADATA_PATH =  (
    Path(__file__).resolve().parent.parent
    / "models" 
    / "metadata.json"
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = mlflow.sklearn.load_model(str(MODEL_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    app.state.threshold = float(metadata["threshold"])

    yield

app = FastAPI(
    title="Telco Customer Churn Prediction",
    description="API to predict customer churn",
    version="1.0.0",
    lifespan=lifespan
)

class CustomerRequest(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]
    OnlineSecurity: Literal[
        "Yes", "No", "No internet service"
    ]
    OnlineBackup: Literal[
        "Yes", "No", "No internet service"
    ]
    DeviceProtection: Literal[
        "Yes", "No", "No internet service"
    ]
    TechSupport: Literal[
        "Yes", "No", "No internet service"
    ]
    StreamingTV: Literal[
        "Yes", "No", "No internet service"
    ]
    StreamingMovies: Literal[
        "Yes", "No", "No internet service"
    ]

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)



@app.post("/predict")
def predict(customer: CustomerRequest):
    
    return predict_customer(
        customer_data = customer.model_dump(),
        model=app.state.model,
        threshold=app.state.threshold
    )

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}