import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
st.title("Gym Member Exercise Data Analysis")
st.write("Analyzing data and predicting calories burned based on user inputs")

# Load data
data_file = "gym_data.csv"  # Update this to the correct file path
try:
    gym = pd.read_csv(data_file)
    st.write("Data loaded successfully.")
except FileNotFoundError:
    st.error(f"Data file '{data_file}' not found. Please check the file path and try again.")
    st.stop()  # Stop the app if data can't be loaded

# Define target variable and features
X = gym.drop('Calories_Burned', axis=1)
y = gym['Calories_Burned']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User Input for Prediction
st.subheader("Enter Your Details for Prediction")
age = st.slider('Age', 18, 100, 25)
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
session_duration = st.slider('Session Duration (hours)', 0.5, 3.0, 1.0)
workout_type = st.selectbox('Workout Type', ['HIIT', 'Strength', 'Yoga'])

# Process user input into model-compatible DataFrame
user_input = {
    'Age': age,
    'Weight (kg)': weight,
    'Session_Duration (hours)': session_duration,
    'Experience_Level': 0,  # Adjust based on your encoding scheme.
    'Gender_Male': 0,  # Assume one-hot encoding.
    'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
    'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
    'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0,
    # Add other columns with appropriate default values as needed
}

user_input_df = pd.DataFrame(user_input, index=[0])

# Ensure user_input_df matches X_train columns
user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)

# Model Selection and Training
selected_model_name = st.selectbox("Select Model for Prediction", ["Linear Regression", "Decision Tree", "Random Forest"])
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}
selected_model = models[selected_model_name]
selected_model.fit(X_train, y_train)

# Make Prediction
user_prediction = selected_model.predict(user_input_df)[0]

# Display Prediction
st.subheader("Model Predictions")
st.write(f"Predicted Calories Burned: {user_prediction:.2f} kcal")

# Display factors contributing to the prediction
st.subheader("Factors Contributing to the Prediction")
st.write(f"**Age:** {age}")
st.write(f"**Weight:** {weight} kg")
st.write(f"**Session Duration:** {session_duration} hours")
st.write(f"**Workout Type:** {workout_type}")
