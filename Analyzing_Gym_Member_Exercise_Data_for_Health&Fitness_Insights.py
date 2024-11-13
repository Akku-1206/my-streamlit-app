#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to load data safely from uploaded file
def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Streamlit application to interact with users
st.title("Health & Fitness Tracker")

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset from uploaded file
    data = load_data(uploaded_file)

    if data is not None:
        # Display basic info and the first few rows of the dataset
        st.write("Dataset Information:")
        st.write(data.info())
        st.write("First few rows of the dataset:")
        st.write(data.head())

        # Check if required columns exist in the dataset
        required_columns = ['Calories_Burned', 'Gender', 'Workout_Type', 'Age', 'Weight (kg)', 'Session_Duration (hours)', 'BMI']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            st.error(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
        else:
            # Preprocessing: Separate features into categorical and numerical
            categorical_features = ['Gender', 'Workout_Type']
            numerical_features = ['Age', 'Weight (kg)', 'Session_Duration (hours)']

            # Define Column Transformer for encoding and scaling
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ]
            )

            # Define features and target for regression model (Calories Burned)
            X = data[numerical_features + categorical_features]  # Use specified features for prediction
            y = data['Calories_Burned']

            # Split data into train and test sets for regression model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the Random Forest Regressor model
            regressor = RandomForestRegressor(random_state=42)
            regressor.fit(preprocessor.fit_transform(X_train), y_train)

            # Make predictions and evaluate the model for regression
            y_pred = regressor.predict(preprocessor.transform(X_test))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f'Mean Absolute Error (Calories Burned): {mae}')
            st.write(f'R-squared (Calories Burned): {r2}')

            # Input sliders for user predictions on Calories Burned
            age = st.slider("Age", 18, 80, 25)
            weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
            workout_type = st.selectbox("Workout Type", options=data['Workout_Type'].unique())
            session_duration = st.slider("Session Duration (hours)", 0.0, 5.0, 1.0)

            if st.button("Predict Calories Burned"):
                user_input = pd.DataFrame([[age, weight, session_duration, 'Male' if workout_type == 'Male' else 'Female', workout_type]], 
                                           columns=['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Gender', 'Workout_Type'])
                
                # Apply preprocessing to user input before prediction.
                user_input_processed = preprocessor.transform(user_input)  
                
                prediction_calories = regressor.predict(user_input_processed)
                st.write(f"Predicted Calories Burned: {prediction_calories[0]:.2f}")

else:
    st.info("Please upload a CSV file to get started.")
