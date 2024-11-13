import streamlit as st
import pandas as pd
import joblib

# Load models
regressor = joblib.load('regressor_model.pkl')
classifier = joblib.load('classifier_model.pkl')

st.title("Health & Fitness Tracker")

# Data upload
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", input_data.head())

# Input sliders
age = st.slider("Age", 18, 80, 25)
weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
session_duration = st.slider("Session Duration (hours)", 0.0, 5.0, 1.0)

# Display Predictions
if st.button("Predict Calories Burned"):
    user_input = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight', 'Session_Duration'])
    prediction = regressor.predict(user_input)
    st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")

if st.button("Classify Experience Level"):
    user_input = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight', 'Session_Duration'])
    prediction = classifier.predict(user_input)
    st.write(f"Predicted Experience Level: {prediction[0]}")
