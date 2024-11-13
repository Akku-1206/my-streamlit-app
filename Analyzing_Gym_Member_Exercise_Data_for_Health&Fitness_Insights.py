#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import streamlit as st
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
        required_columns = ['Calories_Burned', 'Gender', 'Workout_Type', 'Age', 'Weight (kg)', 'Session_Duration (hours)', 'BMI', 'Experience_Level']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            st.error(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
        else:
            # Preprocessing: Separate features into categorical and numerical
            categorical_features = ['Gender', 'Workout_Type']
            numerical_features = [col for col in data.columns if col not in categorical_features + ['Calories_Burned', 'Experience_Level']]

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

            # Check if Calories_Burned is still present after processing (it should not be since we drop it)
            if 'Calories_Burned' not in data.columns:
                st.error("The column 'Calories_Burned' is missing from the original dataset.")
            else:
                # Define features and target for regression model (Calories Burned)
                X_regression = processed_data_df.drop(columns=['Calories_Burned'])
                y_regression = data['Calories_Burned']  # Ensure this column exists in your original data

                # Split data into train and test sets for regression model
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

                # Initialize and train the Random Forest Regressor model
                regressor = RandomForestRegressor(random_state=42)
                regressor.fit(X_train_reg, y_train_reg)

                # Make predictions and evaluate the model for regression
                y_pred_reg = regressor.predict(X_test_reg)
                mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
                r2_reg = r2_score(y_test_reg, y_pred_reg)

                st.write(f'Mean Absolute Error (Calories Burned): {mae_reg}')
                st.write(f'R-squared (Calories Burned): {r2_reg}')

                # Input sliders for user predictions on Calories Burned
                age = st.slider("Age", 18, 80, 25)
                weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
                session_duration = st.slider("Session Duration (hours)", 0.0, 5.0, 1.0)

                if st.button("Predict Calories Burned"):
                    user_input = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight (kg)', 'Session_Duration (hours)'])
                    prediction_calories = regressor.predict(user_input)
                    st.write(f"Predicted Calories Burned: {prediction_calories[0]:.2f}")

                # Define features and target for classification model (Experience Level)
                X_classification = data[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'BMI']]  # Adjust these columns based on your dataset.
                
                y_classification = data['Experience_Level']  # Ensure this column exists in your original data

                label_encoder = LabelEncoder()
                y_encoded_classification = label_encoder.fit_transform(y_classification)

                # Split data into train and test sets for classification model
                X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_encoded_classification, test_size=0.2, random_state=42)

                # Initialize and train the Random Forest Classifier model
                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                classifier.fit(X_train_class, y_train_class)

                if st.button("Classify Experience Level"):
                    user_input_classification = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight (kg)', 'Session_Duration (hours)'])
                    prediction_experience_level = classifier.predict(user_input_classification)
                    experience_level_label = label_encoder.inverse_transform(prediction_experience_level)  # Convert back to original labels.
                    st.write(f"Predicted Experience Level: {experience_level_label[0]}")

else:
    st.info("Please upload a CSV file to get started.")
