"""
api.py
------
FastAPI service that serves the churn model and logs every prediction to
logs/predictions.jsonl (inputs, probability, latency, status) for monitoring.

Where the model comes from:
  1. If the folder MODEL_DIR exists (default: models/churn-model, created by
     mlops/export_model.py) it is loaded from there. This is what Docker uses.
  2. Otherwise the latest registered 'churn-model' version is loaded from the
     local MLflow registry (mlflow.db). This is the local development path.

Run locally from the project root:
    /opt/anaconda3/bin/uvicorn mlops.api:app --reload
Test page:
    http://localhost:8000/docs
"""
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTERED_MODEL_NAME = "churn-model"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models" / "churn-model"))
LOG_PATH = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs")) / "predictions.jsonl"
FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value",
            "tenure_days", "orders_last_90d"]

state = {"model": None, "version": None, "source": None}


def load_model_into_state() -> None:
    if MODEL_DIR.exists():
        state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
        version_file = MODEL_DIR / "model_version.txt"
        state["version"] = int(version_file.read_text().strip()) if version_file.exists() else None
        state["source"] = "exported folder"
    else:
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        latest = max(int(v.version) for v in versions)
        state["model"] = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest}")
        state["version"] = latest
        state["source"] = "mlflow registry"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_into_state()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    recency_days: float = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=0, description="Number of distinct orders")
    monetary: float = Field(..., ge=0, description="Total revenue from the customer")
    avg_order_value: float = Field(..., ge=0)
    tenure_days: float = Field(..., ge=0, description="Days since first purchase")
    orders_last_90d: float = Field(..., ge=0)


def log_prediction(record: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok" if state["model"] is not None else "model_not_loaded",
            "model": REGISTERED_MODEL_NAME, "version": state["version"],
            "loaded_from": state["source"]}


@app.post("/predict")
def predict(payload: CustomerFeatures):
    started = time.perf_counter()
    inputs = payload.model_dump()
    try:
        df = pd.DataFrame([inputs])[FEATURES]
        probability = float(state["model"].predict_proba(df)[0, 1])
        label = int(probability >= 0.5)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        log_prediction({"timestamp": datetime.now(timezone.utc).isoformat(),
                        "inputs": inputs, "churn_probability": probability,
                        "prediction": label, "model_version": state["version"],
                        "latency_ms": latency_ms, "status": "success"})
        return {"churn_probability": round(probability, 4), "churn_prediction": label,
                "model_version": state["version"]}
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        log_prediction({"timestamp": datetime.now(timezone.utc).isoformat(),
                        "inputs": inputs, "latency_ms": latency_ms,
                        "status": "failure", "error": str(exc)})
        raise HTTPException(status_code=500, detail="Prediction failed")
