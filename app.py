import streamlit as st
import numpy as np
import joblib

# Load saved files
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("life_expectancy_model.pkl")

st.title("Life Expectancy Prediction")

st.write("Enter the country health indicators")

# Input fields
Population = st.number_input("Population")
Polio = st.number_input("Polio (%)")
Total_expenditure = st.number_input("Total expenditure")
percentage_expenditure = st.number_input("Percentage expenditure")
infant_deaths = st.number_input("Infant deaths")
Alcohol = st.number_input("Alcohol consumption")
Diphtheria = st.number_input("Diphtheria (%)")
Year = st.number_input("Year")
BMI = st.number_input("BMI")
Schooling = st.number_input("Schooling years")
thinness_5_9 = st.number_input("Thinness 5-9 years")
under_five_deaths = st.number_input("Under-five deaths")
Income_comp = st.number_input("Income composition of resources")
Adult_Mortality = st.number_input("Adult Mortality")
HIV_AIDS = st.number_input("HIV/AIDS")

if st.button("Predict Life Expectancy"):

    input_data = np.array([[Population, Polio, Total_expenditure,
    percentage_expenditure, infant_deaths, Alcohol,
    Diphtheria, Year, BMI, Schooling,
    thinness_5_9, under_five_deaths,
    Income_comp, Adult_Mortality, HIV_AIDS]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Life Expectancy: {prediction[0]:.2f} years")