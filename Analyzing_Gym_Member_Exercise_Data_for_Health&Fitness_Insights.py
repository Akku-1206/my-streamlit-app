#!/usr/bin/env python
# coding: utf-8

# Load essential libraries
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
import os

# Ignore warnings
warnings.filterwarnings('ignore')

# Load dataset and handle file not found error
file_path = "gym_members_exercise_tracking.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found. Please check the file path.")
else:
    gym = pd.read_csv(file_path)

# Display data types and first few rows
print("Data Types:\n", gym.dtypes)
print("First few rows:\n", gym.head())

# List of continuous features for conversion
continuous_features = [
    'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
    'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 
    'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 
    'Experience_Level', 'BMI'
]

# Convert columns to numeric and handle non-numeric values
for col in continuous_features:
    gym[col] = pd.to_numeric(gym[col], errors='coerce')

# Fill missing values with the mean for each column
gym[continuous_features] = gym[continuous_features].fillna(gym[continuous_features].mean())

# Label encode 'Experience_Level' column
le = LabelEncoder()
gym['Experience_Level'] = le.fit_transform(gym['Experience_Level'])

# One-hot encode categorical features
gym = pd.get_dummies(gym, columns=['Gender', 'Workout_Type'], drop_first=True)

# Standardize continuous features
scaler = StandardScaler()
gym[continuous_features] = scaler.fit_transform(gym[continuous_features])

# Define EDA function
def perform_eda(gym):
    """
    Perform exploratory data analysis on the gym members dataset.
    This includes distribution plots, box plots, and correlation analysis.
    """
    sns.set_style('whitegrid')  # Apply seaborn's style

    # Distribution plots for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Key Metrics', fontsize=16)

    sns.histplot(gym['Calories_Burned'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Calories Burned')

    sns.histplot(gym['BMI'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of BMI')

    sns.histplot(gym['Fat_Percentage'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Fat Percentage')

    sns.histplot(gym['Water_Intake (liters)'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Water Intake')

    plt.tight_layout()
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = gym.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

# Call the EDA function
perform_eda(gym)

# Melt one-hot encoded workout type columns back to single column
workout_columns = ['Workout_Type_HIIT', 'Workout_Type_Strength', 'Workout_Type_Yoga']
gym_melted = gym.melt(id_vars=['Calories_Burned'], value_vars=workout_columns, 
                      var_name='Workout_Type', value_name='Performed')

# Filter only rows where workout type was performed
gym_melted = gym_melted[gym_melted['Performed'] == 1].drop(columns='Performed')
gym_melted['Workout_Type'] = gym_melted['Workout_Type'].str.replace('Workout_Type_', '')

# Box plot for calories burned by workout type
plt.figure(figsize=(15, 6))
sns.boxplot(x='Workout_Type', y='Calories_Burned', data=gym_melted)
plt.title('Calories Burned by Workout Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Prepare data for model training
def prepare_data(gym):
    X = gym.drop('Calories_Burned', axis=1)
    y = gym['Calories_Burned']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation function for models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Main function to train and evaluate models
def main(gym):
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(gym)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # List to store model results
    results = []

    # Train and evaluate models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)  # Train model
        y_pred = model.predict(X_test)  # Predictions
        eval_metrics = evaluate_model(y_test, y_pred, name)  # Evaluate model
        results.append(eval_metrics)  # Append results

    # Display performance comparison
    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(results_df.round(4))

# Execute main function
main(gym)
