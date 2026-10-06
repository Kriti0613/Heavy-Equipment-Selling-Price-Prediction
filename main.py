"""FastAPI service for the heavy-equipment selling-price model."""

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent / "model_bundle.pkl"))


class PredictionRequest(BaseModel):
    AssetAge: float = Field(allow_inf_nan=False)
    OperationalHoursMeter: float = Field(allow_inf_nan=False)
    ManufactureYear: float = Field(allow_inf_nan=False)


class PredictionResponse(BaseModel):
    selling_price: float


def load_model_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_bundle()
    yield


app = FastAPI(
    title="Heavy Equipment Machinery Price Predictor",
    lifespan=lifespan,
)


def make_features(payload: PredictionRequest, bundle: dict[str, Any]) -> pd.DataFrame:
    columns = bundle["feature_columns"]
    values = {column: 0 for column in columns}

    for column, value in zip(
        bundle["numerical_columns"],
        bundle["numerical_imputer"].statistics_,
    ):
        values[column] = value

    for column, value in bundle.get("categorical_encoded_defaults", {}).items():
        values[column] = value

    values.update(payload.model_dump())
    return pd.DataFrame([[values[column] for column in columns]], columns=columns)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heavy equipment price prediction API. Use /docs to try it."}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    bundle = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle is unavailable.")

    features = make_features(payload, bundle)
    log_price = float(bundle["model"].predict(features)[0])
    price = math.expm1(log_price)
    if not math.isfinite(price):
        raise HTTPException(status_code=500, detail="The model returned an invalid prediction.")
    return PredictionResponse(selling_price=max(0.0, price))
