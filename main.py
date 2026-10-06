"""FastAPI service for the notebook's heavy-equipment price model."""

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent / "model_bundle.pkl"))


class PredictionRequest(BaseModel):
    features: dict[str, str | int | float | None] = Field(
        description="Raw equipment feature values from the training dataset."
    )


class PredictionResponse(BaseModel):
    selling_price: float


def load_model_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        logger.warning("Model bundle not found at %s", MODEL_PATH)
        return None

    bundle = joblib.load(MODEL_PATH)
    required_keys = {
        "model",
        "feature_columns",
        "numerical_columns",
        "categorical_columns",
        "numerical_imputer",
        "categorical_imputer",
        "categorical_encoder",
    }
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise RuntimeError(
            f"Unsupported model bundle at {MODEL_PATH}; rerun the notebook's export cell."
        )
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_bundle = load_model_bundle()
    yield


app = FastAPI(
    title="Heavy Equipment Machinery Price Predictor",
    description="Predicts machinery selling prices using the model trained in the notebook.",
    version="1.0.0",
    lifespan=lifespan,
)


def prepare_features(
    values: dict[str, str | int | float | None],
    bundle: dict[str, Any],
) -> pd.DataFrame:
    feature_columns: list[str] = bundle["feature_columns"]
    numerical_columns: list[str] = bundle["numerical_columns"]
    categorical_columns: list[str] = bundle["categorical_columns"]
    allowed_columns = set(feature_columns) | {"TransactionDate"}
    unexpected = set(values) - allowed_columns
    if unexpected:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown feature(s): {', '.join(sorted(unexpected))}",
        )

    row = {column: values.get(column) for column in feature_columns}

    if values.get("TransactionDate") is not None:
        date_value = values["TransactionDate"]
        if not isinstance(date_value, str):
            raise HTTPException(status_code=422, detail="TransactionDate must be a date string.")
        transaction_date = pd.to_datetime(date_value, errors="coerce")
        if pd.isna(transaction_date):
            raise HTTPException(status_code=422, detail="TransactionDate must be a valid date.")

        for column, value in (
            ("TransactionYear", transaction_date.year),
            ("TransactionMonth", transaction_date.month),
            ("TransactionQuarter", transaction_date.quarter),
        ):
            if column in row:
                row[column] = value

        manufacture_year = values.get("ManufactureYear")
        if manufacture_year is not None and "AssetAge" in row:
            try:
                row["AssetAge"] = transaction_date.year - float(manufacture_year)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=422,
                    detail="ManufactureYear must be numeric.",
                ) from None

    engineered_values = {
        "HasOperationalHours": int(values.get("OperationalHoursMeter") is not None),
        "HasVariantModifier": int(values.get("Spec_VariantModifier") is not None),
        "DescriptorLength": len(str(values["Spec_FullDescriptor"]))
        if values.get("Spec_FullDescriptor") is not None
        else 0,
    }
    for column, value in engineered_values.items():
        if column in row:
            row[column] = value

    frame = pd.DataFrame([row], columns=feature_columns)
    for column in numerical_columns:
        value = frame.at[0, column]
        if value is None or pd.isna(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"{column} must be numeric.") from None
        if not math.isfinite(number):
            raise HTTPException(
                status_code=422,
                detail=f"{column} must be a finite number or null.",
            )
        frame.at[0, column] = number

    if numerical_columns:
        transformed = bundle["numerical_imputer"].transform(frame[numerical_columns])
        for index, column in enumerate(numerical_columns):
            frame[column] = transformed[:, index]

    if categorical_columns:
        for column in categorical_columns:
            value = frame.at[0, column]
            if value is not None and not pd.isna(value):
                frame[column] = str(value)
        categorical_values = bundle["categorical_imputer"].transform(
            frame[categorical_columns]
        )
        encoded = bundle["categorical_encoder"].transform(categorical_values)
        for index, column in enumerate(categorical_columns):
            frame[column] = encoded[:, index]

    return frame[feature_columns]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heavy equipment price prediction API. See /docs for usage."}


@app.get("/health")
def health(request: Request) -> dict[str, bool | str]:
    loaded = request.app.state.model_bundle is not None
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    bundle = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model bundle is unavailable. Export model_bundle.pkl from the notebook.",
        )

    features = prepare_features(payload.features, bundle)
    log_price = float(bundle["model"].predict(features)[0])
    price = math.expm1(log_price)
    if not math.isfinite(price):
        raise HTTPException(status_code=500, detail="The model returned an invalid prediction.")
    return PredictionResponse(selling_price=max(0.0, price))
