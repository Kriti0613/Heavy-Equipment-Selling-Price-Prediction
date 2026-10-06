# Heavy Equipment Analysis & Prediction

An end-to-end machine learning project focused on analysing heavy equipment transaction data and developing predictive models for equipment-related financial outcomes.

## Overview

This project explores a large, multi-dimensional dataset containing transactional, operational, geographic, and technical information associated with heavy industrial equipment.

The dataset combines numerical variables, categorical attributes, dates, geographic indicators, and high-cardinality technical features. The project focuses on understanding the underlying patterns in the data, identifying important factors associated with equipment value and transactions, and evaluating machine learning approaches for predictive modelling.

## Objectives

* Perform exploratory data analysis to understand transaction and equipment characteristics.
* Analyse relationships between equipment specifications, operational attributes, and financial outcomes.
* Handle missing values, categorical variables, dates, and high-cardinality features.
* Engineer meaningful features from the available transactional and technical data.
* Develop and evaluate machine learning models for predicting the target variable.
* Compare model performance and identify the features contributing most to predictions.

## Dataset

The dataset contains records representing finalized accounting or operational events related to heavy equipment. Each record includes a combination of:

* Transaction and accounting information
* Equipment specifications
* Mechanical and technical attributes
* Geographic indicators
* Operational characteristics
* Date-related features
* Financial variables

## Project Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Feature encoding and transformation
6. Model development
7. Hyperparameter tuning
8. Model evaluation
9. Feature importance and interpretation

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## FastAPI prediction service

The API serves the LightGBM regressor trained in the notebook. Run the notebook through its final model export cell first; it writes `model_bundle.pkl` to the project root with the model and its fitted preprocessing objects.

Install the project dependencies and start the service locally from the project root:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

For Render, create a **Web Service** for this repository. Use `pip install -r requirements.txt` as the build command and `uvicorn main:app --host 0.0.0.0 --port $PORT` as the start command. The included `Procfile` contains the same start command. Add the exported `model_bundle.pkl` as a secret file at the project root, or set the `MODEL_PATH` environment variable to its mounted path. Export the bundle from the notebook before deploying; `model.pkl` is not compatible.

Open `http://127.0.0.1:8000/predict` locally (or your deployed service's `/predict`) to enter prediction values in the browser form. The **Predict** button submits them to `POST /predict` and displays the selling price. For interactive API documentation or direct JSON requests, use `http://127.0.0.1:8000/docs` (or your deployed service's `/docs`):

```json
{
  "AssetAge": 2,
  "OperationalHoursMeter": 250,
  "ManufactureYear": 2016
}
```

The other model features use saved training defaults. The response contains the predicted selling price in dollars:

```json
{
  "selling_price": 28500.0
}
```

If the model bundle is unavailable, `/predict` returns HTTP 503.
