# Crop Production Prediction (India)
This repository contains a web application for predicting crop production based on state, district, crop, season, year, and area of cultivation. The prediction model uses machine learning techniques and is implemented using Streamlit for the user interface and Jupyter Notebook for model training.

## Features
Dynamic User Input: Users can select state, district, crop, season, and specify year and area for prediction.
Interactive Interface: Built with Streamlit, providing a seamless and intuitive UI for predictions.
Machine Learning Model: The predictive model is trained using historical data on crop production in India and incorporates feature scaling and label encoding.
Preprocessed Data: Includes label encoders and scalers to handle categorical data and normalize numerical features.
Accurate Predictions: The model utilizes advanced regression techniques for precise crop production estimation.

## How It Works
User Input: Users provide the following inputs:

State Name

District Name

Crop Type

Season

Year of cultivation

Cultivated Area (in hectares)

## Data Transformation:

Inputs are encoded using pre-trained label encoders for categorical features.
Data is scaled using a pre-trained scaler to ensure compatibility with the model.
## Prediction:

The processed input is fed into the trained machine learning model, which outputs the predicted crop production.
## Technology Stack
Frontend: Streamlit

Backend: Python

Libraries:

pandas for data manipulation

joblib for model persistence and loading pre-trained components

scikit-learn for machine learning

Model Training: Jupyter Notebook

## Clone the repository:

git clone https://github.com/Saivardhan190/crop-production-prediction.git

Navigate to the project directory:

cd crop-production-prediction

Install the required Python packages:

pip install -r requirements.txt

Place the following files in the project directory:

Crop Production data.csv (dataset)

production.pkl (trained machine learning model)

feature_names.pkl (list of feature names)

scaler.pkl (scaler for feature scaling)

le_state.pkl, le_district.pkl, le_season.pkl, le_crop.pkl (label encoders)

Run the Streamlit application:

streamlit run app.py

## Dataset
The dataset used for this project includes historical crop production data from various states and districts in India, sourced from the Ministry of Agriculture and Farmers Welfare. It covers information on seasons, crops, and cultivated area.

## Model Details
The predictive model was trained using regression techniques, leveraging features such as:

State and district names (encoded)
Season and crop type (encoded)
Year and area of cultivation
The model was built and validated using Jupyter Notebook and then exported as a .pkl file for deployment.

## Usage
Start the application using Streamlit.

Select or enter the required inputs (state, district, crop, season, year, and area).

Click the Predict button to view the predicted crop production.

## Future Enhancements
Additional Features: Incorporating weather and soil data for improved accuracy.

Advanced Models: Using deep learning or ensemble methods for better predictions.

Data Visualization: Adding charts and graphs for better insight into crop production trends.
## License
This project is open-source and available under the MIT License.
