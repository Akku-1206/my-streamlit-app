import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# App title and description
st.title("Gym Member Exercise Data Analysis")
st.write("Analyzing data and predicting calories burned based on user inputs")

# File upload option
uploaded_file = st.file_uploader("Upload your gym data CSV file", type="csv")
if uploaded_file is not None:
    gym = pd.read_csv(uploaded_file)
    st.write("Data loaded successfully.")
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# Define target variable and features
X = gym.drop('Calories_Burned', axis=1)
y = gym['Calories_Burned']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User input section
st.subheader("Enter Your Details for Prediction")
age = st.slider('Age', 18, 100, 25)
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
session_duration = st.slider('Session Duration (hours)', 0.5, 3.0, 1.0)
workout_type = st.selectbox('Workout Type', ['HIIT', 'Strength', 'Yoga'])

# Process user input into DataFrame
user_input = {
    'Age': age,
    'Weight (kg)': weight,
    'Session_Duration (hours)': session_duration,
    'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
    'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
    'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0,
}
user_input_df = pd.DataFrame(user_input, index=[0])

# Match user input to X_train columns
user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)

# Model selection and training
selected_model_name = st.selectbox("Select Model for Prediction", ["Linear Regression", "Decision Tree", "Random Forest"])
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}
selected_model = models[selected_model_name]
selected_model.fit(X_train, y_train)

# Make and display prediction
user_prediction = selected_model.predict(user_input_df)[0]
st.subheader("Model Predictions")
st.write(f"Predicted Calories Burned: {user_prediction:.2f} kcal")
