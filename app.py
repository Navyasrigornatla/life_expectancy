import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load preprocessor and trained model
preprocessor = joblib.load("preprocessor.pkl")  # if you used a preprocessor pipeline
model = joblib.load("life_expectancy_model.pkl")  # your trained stacking model

st.title("Life Expectancy Prediction")

st.markdown("Enter the values for the following features:")

# Create input fields for all 15 features
population = st.number_input("Population", min_value=0, value=1000000)
polio = st.number_input("Polio immunization rate", min_value=0, max_value=100, value=95)
total_expenditure = st.number_input("Total expenditure (per capita)", value=5.5)
percentage_expenditure = st.number_input("Percentage expenditure on health", value=3.2)
infant_deaths = st.number_input("Infant deaths", min_value=0, value=10)
alcohol = st.number_input("Alcohol consumption (per capita)", value=4.0)
diphtheria = st.number_input("Diphtheria immunization rate", min_value=0, max_value=100, value=97)
year = st.number_input("Year", min_value=2000, max_value=2030, value=2016)
bmi = st.number_input("BMI", value=24.5)
schooling = st.number_input("Schooling (years)", min_value=0.0, value=12.0)
thinness = st.number_input("Thinness 5-9 years (%)", value=1.5)
under_five_deaths = st.number_input("Under-five deaths", min_value=0, value=3)
income_composition = st.number_input("Income composition of resources", min_value=0.0, max_value=1.0, value=0.8)
adult_mortality = st.number_input("Adult Mortality", min_value=0, value=150)
hiv_aids = st.number_input("HIV/AIDS prevalence", min_value=0.0, value=0.1)

# Button to predict
if st.button("Predict Life Expectancy"):
    # Create input array in the correct order
    input_data = np.array([[
        population, polio, total_expenditure, percentage_expenditure,
        infant_deaths, alcohol, diphtheria, year, bmi, schooling,
        thinness, under_five_deaths, income_composition, adult_mortality, hiv_aids
    ]])
    
    # If you have a preprocessor pipeline
    input_processed = preprocessor.transform(input_data)
    
    # Predict
    prediction = model.predict(input_processed)
    
    st.success(f"Predicted Life Expectancy: {prediction[0]:.2f} years")
