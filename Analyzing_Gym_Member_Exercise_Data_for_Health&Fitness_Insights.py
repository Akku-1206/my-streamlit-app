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

# Load dataset
file_path = "gym_members_exercise_tracking.csv"
try:
    gym = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"{file_path} not found. Please check the file path.")
    st.stop()

# Display the first few rows
st.title("Gym Members Exercise Tracking Analysis")
st.subheader("Dataset Preview")
st.write(gym.head())

# Data preprocessing
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

# EDA
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

# Model Training
st.subheader("Model Training and Evaluation")

# Function for splitting data
def prepare_data(gym):
    X = gym.drop('Calories_Burned', axis=1)
    y = gym['Calories_Burned']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(gym)

# Model Training
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Training and displaying results
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval_metrics = evaluate_model(y_test, y_pred)
    eval_metrics['Model'] = name
    results.append(eval_metrics)

# Display results
st.write("### Model Performance Comparison")
results_df = pd.DataFrame(results)
st.write(results_df)
