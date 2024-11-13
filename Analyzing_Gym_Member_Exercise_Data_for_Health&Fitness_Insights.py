#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# Function to load data safely
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please check the path.")
        return None

# Load the dataset (update this path)
file_path = r"C:\Users\ASUS\OneDrive\Desktop\gym_members_exercise_tracking.csv"
data = load_data(file_path)

if data is not None:
    # Display basic info and the first few rows of the dataset
    st.write("Dataset Information:")
    st.write(data.info())
    st.write("First few rows of the dataset:")
    st.write(data.head())

    # Preprocessing: Separate features into categorical and numerical
    categorical_features = ['Gender', 'Workout_Type']
    numerical_features = [col for col in data.columns if col not in categorical_features]

    # Define Column Transformer for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    # Apply transformations to the dataset
    processed_data = preprocessor.fit_transform(data)

    # Create a DataFrame to view the processed data
    processed_data_df = pd.DataFrame(processed_data, columns=numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

    # Define features and target for regression model
    X = processed_data_df.drop(columns=['Calories_Burned'])
    y = processed_data_df['Calories_Burned']

    # Split data into train and test sets for regression model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor model
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f'Mean Absolute Error: {mae}')
    st.write(f'R-squared: {r2}')

    # Save the trained model to a .pkl file (optional)
    joblib.dump(regressor, 'regressor_model.pkl')

    # Streamlit application to interact with users
    st.title("Health & Fitness Tracker")

    uploaded_file = st.file_uploader("Choose a CSV file")
    
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data", input_data.head())

        age = st.slider("Age", 18, 80, 25)
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
        session_duration = st.slider("Session Duration (hours)", 0.0, 5.0, 1.0)

        if st.button("Predict Calories Burned"):
            user_input = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight', 'Session_Duration'])
            prediction = regressor.predict(user_input)
            st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")

        if st.button("Classify Experience Level"):
            user_input_classification = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight', 'Session_Duration'])
            prediction_classification = classifier.predict(user_input_classification)  # Ensure classifier is defined and trained before this point.
            experience_level = label_encoder.inverse_transform(prediction_classification)  # Convert back to original labels.
            st.write(f"Predicted Experience Level: {experience_level[0]}")
else:
    st.error("Data could not be loaded.")
