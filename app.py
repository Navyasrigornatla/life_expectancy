# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

# -----------------------------
# Load your trained model & preprocessor
# -----------------------------
model = joblib.load("life_expectancy_model.pkl")        # XGBoost model
preprocessor = joblib.load("preprocessor.pkl")          # ColumnTransformer for 15 features

# -----------------------------
# Feature mapping
# API-safe names → Model column names
# -----------------------------
feature_map = {
    "Population": "Population",
    "Polio": "Polio",
    "Total_expenditure": "Total expenditure",
    "Percentage_expenditure": "percentage expenditure",
    "Infant_deaths": "infant deaths",
    "Alcohol": "Alcohol",
    "Diphtheria": "Diphtheria",
    "Year": "Year",
    "BMI": "BMI",
    "Schooling": "Schooling",
    "Thinness_5_9_years": "thinness 5-9 years",
    "Under_five_deaths": "under-five deaths",
    "Income_composition_of_resources": "Income composition of resources",
    "Adult_Mortality": "Adult Mortality",
    "HIV_AIDS": "HIV/AIDS"
}

# -----------------------------
# Input schema
# -----------------------------
class InputData(BaseModel):
    Population: float
    Polio: float
    Total_expenditure: float
    Percentage_expenditure: float
    Infant_deaths: float
    Alcohol: float
    Diphtheria: float
    Year: float
    BMI: float
    Schooling: float
    Thinness_5_9_years: float
    Under_five_deaths: float
    Income_composition_of_resources: float
    Adult_Mortality: float
    HIV_AIDS: float

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="Life Expectancy Predictor",
    description="Predict life expectancy using trained ML model (15 features)",
    version="1.0"
)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"message": "API is running. Use /predict to get predictions."}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: List[InputData]):

    rows = []
    for item in data:
        row = {}
        for api_key, model_key in feature_map.items():
            row[model_key] = getattr(item, api_key)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Preprocess input
    X_transformed = preprocessor.transform(df)

    # Make predictions
    predictions = model.predict(X_transformed)

    return {"predictions": predictions.tolist()}

# -----------------------------
# Local run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)