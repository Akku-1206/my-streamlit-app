import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title of the app
st.title("Gym Members Exercise Tracking Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Input Features & Prediction", "Data Visualizations"])

# Load and preprocess data
if "data" not in st.session_state:
    st.session_state["data"] = None

if page == "Data Upload":
    st.subheader("Upload your Gym Member Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state["data"] = data
        st.write("Data Preview:")
        st.write(data.head())

if page == "Input Features & Prediction":
    st.subheader("Enter Your Details for Prediction")

    if st.session_state["data"] is not None:
        data = st.session_state["data"]

        # Feature inputs
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        weight = st.number_input("Weight (kg)", min_value=18, max_value=200, value=70)
        session_duration = st.number_input("Session Duration (hours)", min_value=0.5, max_value=3.0, step=0.1, value=1.0)
        workout_type = st.selectbox("Workout Type", ["HIIT", "Strength", "Yoga"])

        # One-hot encode workout type
        workout_type_encoded = {
            "Workout_Type_HIIT": int(workout_type == "HIIT"),
            "Workout_Type_Strength": int(workout_type == "Strength"),
            "Workout_Type_Yoga": int(workout_type == "Yoga")
        }
        
        # Create DataFrame for the input
        user_data = pd.DataFrame([{
            "Age": age,
            "Weight": weight,
            "Session_Duration": session_duration,
            **workout_type_encoded
        }])

        st.write("Entered Details:")
        st.write(user_data)

        # Prepare data for model
        if 'Calories_Burned' in data.columns:
            # Encode categorical features and split data
            data = pd.get_dummies(data, columns=["Workout_Type"], drop_first=True)
            X = data.drop("Calories_Burned", axis=1)
            y = data["Calories_Burned"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Align user_data with model columns
            user_data = user_data.reindex(columns=X.columns, fill_value=0)
            
            # Prediction
            prediction = model.predict(user_data)
            st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")

            # Option to show evaluation metrics
            y_pred = model.predict(X_test)
            st.write("Model Evaluation:")
            st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")
        else:
            st.write("Calories_Burned column is missing in the uploaded data. Cannot perform prediction.")

if page == "Data Visualizations":
    st.subheader("Data Visualizations")

    if st.session_state["data"] is not None:
        data = st.session_state["data"]
        
        st.write("Feature Distributions:")
        numeric_cols = data.select_dtypes(include=np.number).columns
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data[numeric_cols], kde=True, ax=ax)
        st.pyplot(fig)

        if 'Calories_Burned' in data.columns:
            st.write("Correlation Heatmap:")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
    else:
        st.write("Please upload data first.")
