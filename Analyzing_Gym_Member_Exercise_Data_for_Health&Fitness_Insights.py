#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the essential libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Load dataset 
gym = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\gym_members_exercise_tracking.csv")

# Display data types and first few rows
print("Data Types:\n", gym.dtypes)
print("First few rows:\n", gym.head())


# In[3]:


# List of continuous features for conversion
continuous_features = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
                       'Resting_BPM', 'Session_Duration (hours)', 
                       'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)', 
                       'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']

# Convert columns to numeric and handle non-numeric values
for col in continuous_features:
    gym[col] = pd.to_numeric(gym[col], errors='coerce')

# Fill missing values with the mean for each column
gym[continuous_features] = gym[continuous_features].fillna(gym[continuous_features].mean())


# In[4]:


# Label encoding column 'Experience_Level'
le = LabelEncoder()
gym['Experience_Level'] = le.fit_transform(gym['Experience_Level'])

# One-hot encoding 'Gender' and 'Workout_Type' column
gym = pd.get_dummies(gym, columns=['Gender', 'Workout_Type'], drop_first=True)


# In[5]:


scaler = StandardScaler()
gym[continuous_features] = scaler.fit_transform(gym[continuous_features])


# In[6]:


def perform_eda(gym):
    """
    Perform exploratory data analysis on the gym members dataset.
    This includes distribution plots, box plots, and correlation analysis.
    """
    plt.style.use('seaborn')

    # Create a figure for distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Key Metrics', fontsize=16)

    # Plot distributions of key metrics
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


    # Correlation analysis
    plt.figure(figsize=(12, 8))
    correlation_matrix = gym.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

# Call the EDA function
perform_eda(gym)


# In[7]:


# Melt the one-hot encoded workout type columns back into a single column
workout_columns = ['Workout_Type_HIIT', 'Workout_Type_Strength', 'Workout_Type_Yoga']
gym_melted = gym.melt(id_vars=['Calories_Burned'], value_vars=workout_columns, 
                      var_name='Workout_Type', value_name='Performed')

# Filter to keep only rows where the workout type was performed
gym_melted = gym_melted[gym_melted['Performed'] == 1]

# Remove the "Performed" column since it's no longer needed
gym_melted = gym_melted.drop(columns='Performed')

# Clean up the `Workout_Type` names to remove prefixes (optional)
gym_melted['Workout_Type'] = gym_melted['Workout_Type'].str.replace('Workout_Type_', '')

# Plot the box plot for calories burned by workout type
plt.figure(figsize=(15, 6))
sns.boxplot(x='Workout_Type', y='Calories_Burned', data=gym_melted)
plt.title('Calories Burned by Workout Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# In[8]:


def prepare_data(gym):
    X = gym.drop('Calories_Burned', axis=1)
    y = gym['Calories_Burned']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# In[9]:


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


# In[10]:


def main(gym):
    # Prepare training and test sets
    X_train, X_test, y_train, y_test = prepare_data(gym)

    # Initialize models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model on training data
        model.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = model.predict(X_test)

        # Evaluate model performance
        eval_metrics = evaluate_model(y_test, y_pred, name)
        results.append(eval_metrics)

    # Display performance comparison
    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(results_df.round(4))

# Call the main function with the gym dataset
main(gym)


# In[ ]:




