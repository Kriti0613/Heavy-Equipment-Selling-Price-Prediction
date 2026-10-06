"""FastAPI service for the heavy-equipment selling-price model."""

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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
    return {"message": "Heavy equipment price prediction API. Use /predict for the form or /docs for the API."}


@app.get("/predict", response_class=HTMLResponse)
def predict_form() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Heavy Equipment Price Prediction</title>
        <style>
          body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 36rem; padding: 0 1rem; }
          label { display: block; font-weight: 600; margin-top: 1rem; }
          input, button { box-sizing: border-box; font: inherit; margin-top: 0.35rem; padding: 0.6rem; width: 100%; }
          button { cursor: pointer; margin-top: 1.25rem; }
          #result { margin-top: 1.25rem; }
        </style>
      </head>
      <body>
        <main>
          <h1>Heavy Equipment Price Prediction</h1>
          <form id="prediction-form">
            <label for="asset-age">Asset age</label>
            <input id="asset-age" name="AssetAge" type="number" step="any" required>

            <label for="operational-hours">Operational hours meter</label>
            <input id="operational-hours" name="OperationalHoursMeter" type="number" step="any" required>

            <label for="manufacture-year">Manufacture year</label>
            <input id="manufacture-year" name="ManufactureYear" type="number" step="any" required>

            <button type="submit">Predict</button>
          </form>
          <p id="result" aria-live="polite"></p>
        </main>
        <script>
          const form = document.getElementById("prediction-form");
          const result = document.getElementById("result");

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            result.textContent = "Calculating prediction...";

            const formData = new FormData(form);
            const payload = Object.fromEntries(
              Array.from(formData, ([name, value]) => [name, Number(value)])
            );

            try {
              const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
              });
              const data = await response.json();
              if (!response.ok) {
                throw new Error(data.detail || "Prediction failed.");
              }
              result.textContent = "Predicted selling price: " +
                new Intl.NumberFormat("en-US", {
                  style: "currency",
                  currency: "USD"
                }).format(data.selling_price);
            } catch (error) {
              result.textContent = "Unable to predict: " + error.message;
            }
          });
        </script>
      </body>
    </html>
    """


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
