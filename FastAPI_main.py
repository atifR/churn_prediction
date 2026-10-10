from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

# resolve paths relative to this file, so the app works from any working directory
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "best_churn_model.pkl"
feature_importance_path = BASE_DIR / "models" / "feature_importance.csv"
x_test_path = BASE_DIR / "models" / "x_test.csv"


ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load(model_path)   # load once at startup
    ml["feature_importance"] = pd.read_csv(feature_importance_path)
    ml["x_test"] = pd.read_csv(x_test_path)
    yield
    ml.clear()

app = FastAPI(title="Churn Prediction API", lifespan=lifespan)

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]

class Customer(BaseModel):
    # same 19 columns (names and values) the pipeline was trained on
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    # example shown in the /docs page (first row of models/x_test.csv)
    model_config = ConfigDict(json_schema_extra={"example": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 59,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 75.95,
        "TotalCharges": 4542.35,
    }})

class Prediction(BaseModel):
    churn: bool
    # SVC decision_function output: > 0 means churn, larger magnitude = more confident.
    # It is NOT a probability (the saved SVC was trained without probability=True).
    score: float

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    importance_pct: float

def run_model(df: pd.DataFrame) -> list[Prediction]:
    model = ml["model"]
    labels = model.predict(df)              # "Yes" / "No"
    scores = model.decision_function(df)
    return [
        Prediction(churn=label == "Yes", score=float(score))
        for label, score in zip(labels, scores)
    ]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml}

@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    return run_model(df)[0]

@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    df = pd.DataFrame([c.model_dump() for c in customers])
    return run_model(df)

@app.get("/feature-importance", response_model=list[FeatureImportance])
def feature_importance(top: int = 10):

    df = ml["feature_importance"].head(top)
    return [
        FeatureImportance(
            feature=row.Feature,
            importance=row.Importance,
            importance_pct=row.Importance_pct,
        )
        for row in df.itertuples()
    ]

@app.get("/sample-customers", response_model=list[Customer])
def sample_customers(n: int = Query(5, ge=1, le=100)):
    # random rows from the test set, ready to paste into /predict or /predict/batch
    df = ml["x_test"].sample(n)
    return df.to_dict(orient="records")
