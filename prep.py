import streamlit as st
import pandas as pd
import joblib

data = pd.read_csv("Crop Production data.csv")
states = data['State_Name'].unique()
seasons = data['Season'].unique()
crops = data['Crop'].unique()
model = joblib.load("production.pkl")
feature_names = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Crop Production Prediction")
state = st.selectbox("State_Name", states)
filtered_districts = data[data['State_Name'] == state]['District_Name'].unique()
district = st.selectbox("District_Name", filtered_districts)
season = st.selectbox("Season", seasons)
crop = st.selectbox("Crop", crops)
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
