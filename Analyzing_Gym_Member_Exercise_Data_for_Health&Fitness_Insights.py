import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Function to preprocess the uploaded dataset
def preprocess_data(gym):
    continuous_features = [
        'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 
        'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 
        'Experience_Level', 'BMI'
    ]

    # Convert to numeric and fill missing values
    for col in continuous_features:
        gym[col] = pd.to_numeric(gym[col], errors='coerce')
    gym[continuous_features] = gym[continuous_features].fillna(gym[continuous_features].mean())

    # Label encoding for Experience_Level
    le = LabelEncoder()
    gym['Experience_Level'] = le.fit_transform(gym['Experience_Level'])

    # One-hot encoding for categorical features
    gym = pd.get_dummies(gym, columns=['Gender', 'Workout_Type'], drop_first=True)

    # Standardization of continuous features
    scaler = StandardScaler()
    gym[continuous_features] = scaler.fit_transform(gym[continuous_features])
    
    return gym

# Load and preprocess the initial dataset
file_path = "gym_members_exercise_tracking.csv"
try:
    gym = pd.read_csv(file_path)
    gym = preprocess_data(gym)
except FileNotFoundError:
    st.error(f"{file_path} not found. Please check the file path.")
    st.stop()

# Streamlit User Interface
st.title("Gym Members Exercise Tracking Analysis")
st.subheader("Dataset Preview")
st.write(gym.head())

# Data Upload
st.subheader("Upload Your Own Dataset")
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

if uploaded_file is not None:
    uploaded_gym = pd.read_csv(uploaded_file)
    uploaded_gym = preprocess_data(uploaded_gym)
    st.write(uploaded_gym.head())

# Input features for prediction
st.subheader("Enter Your Details for Prediction")
age = st.slider('Age', 18, 100, 25)
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
session_duration = st.slider('Session Duration (hours)', 0.5, 3.0, 1.0)
workout_type = st.selectbox('Workout Type', ['HIIT', 'Strength', 'Yoga'])

# Data Preprocessing for User Input
user_input = {
    'Age': age,
    'Weight (kg)': weight,
    'Session_Duration (hours)': session_duration,
    'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
    'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
    'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0,
}
user_input_df = pd.DataFrame(user_input, index=[0])

# Model Training
X_train, X_test, y_train, y_test = train_test_split(gym.drop('Calories_Burned', axis=1), gym['Calories_Burned'], test_size=0.2, random_state=42)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })

results_df = pd.DataFrame(results)
st.subheader("Model Performance Comparison")
st.write(results_df)

# Select model for prediction
selected_model_name = st.selectbox("Select Model for Prediction", ["Linear Regression", "Decision Tree", "Random Forest"])
selected_model = models[selected_model_name]

# Make prediction for user input
user_prediction = selected_model.predict(user_input_df)[0]
st.subheader(f"Predicted Calories Burned: {user_prediction:.2f} kcal")

# Explanation of factors contributing to the prediction
st.subheader("Factors Contributing to the Prediction")
st.write(f"Age: {age}")
st.write(f"Weight: {weight} kg")
st.write(f"Session Duration: {session_duration} hours")
st.write(f"Workout Type: {workout_type}")

# Data Visualizations (EDA)
st.subheader("Exploratory Data Analysis")

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.histplot(gym['Calories_Burned'], kde=True, ax=axes[0, 0]).set_title('Calories Burned')
sns.histplot(gym['BMI'], kde=True, ax=axes[0, 1]).set_title('BMI')
sns.histplot(gym['Fat_Percentage'], kde=True, ax=axes[1, 0]).set_title('Fat Percentage')
sns.histplot(gym['Water_Intake (liters)'], kde=True, ax=axes[1, 1]).set_title('Water Intake')
st.pyplot(fig)

# Correlation matrix
st.write("### Correlation Matrix")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(gym.corr(), annot=True, cmap='coolwarm', center=0)
st.pyplot(fig)

# Workout Type vs. Calories Burned
st.write("### Calories Burned by Workout Type")
workout_columns = ['Workout_Type_HIIT', 'Workout_Type_Strength', 'Workout_Type_Yoga']
gym_melted = gym.melt(id_vars=['Calories_Burned'], value_vars=workout_columns, 
                      var_name='Workout_Type', value_name='Performed')
gym_melted = gym_melted[gym_melted['Performed'] == 1].drop(columns='Performed')
gym_melted['Workout_Type'] = gym_melted['Workout_Type'].str.replace('Workout_Type_', '')

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x='Workout_Type', y='Calories_Burned', data=gym_melted)
st.pyplot(fig)
