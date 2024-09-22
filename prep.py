import streamlit as st
import joblib
import pandas as pd

model = joblib.load("production.pkl")
feature_names = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Crop Production Prediction")
state = st.text_input("State Name")
district = st.text_input("District Name")
season = st.text_input("Season")
crop = st.text_input("Crop")
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
area = st.number_input("Area", min_value=0.0, step=0.1)

le_state = joblib.load("le_state.pkl")
le_district = joblib.load("le_district.pkl")
le_season = joblib.load("le_season.pkl")
le_crop = joblib.load("le_crop.pkl")

def transform_input(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return -1 
state_encoded = transform_input(le_state, state)
district_encoded = transform_input(le_district, district)
season_encoded = transform_input(le_season, season)
crop_encoded = transform_input(le_crop, crop)
input_data = pd.DataFrame({
    'State_Name': [state_encoded],
    'District_Name': [district_encoded],
    'Season': [season_encoded],
    'Crop': [crop_encoded],
    'Area': [area],        
    'Crop_Year': [year]    
})
input_data = input_data[feature_names]
input_data_scaled = scaler.transform(input_data)
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Crop Production: {prediction[0]}")
