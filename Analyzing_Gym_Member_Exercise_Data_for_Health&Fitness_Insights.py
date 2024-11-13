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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
file_path = r"C:\Users\ASUS\OneDrive\Desktop\gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

# Display basic info and the first few rows of the dataset
data_info = data.info()
data_head = data.head()

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

# Data Visualization: Distribution of key metrics
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.histplot(processed_data_df['Calories_Burned'], kde=True, color='blue')
plt.title('Distribution of Calories Burned')

plt.subplot(1, 3, 2)
sns.histplot(processed_data_df['BMI'], kde=True, color='green')
plt.title('Distribution of BMI')

plt.subplot(1, 3, 3)
sns.histplot(processed_data_df['Fat_Percentage'], kde=True, color='purple')
plt.title('Distribution of Fat Percentage')

plt.tight_layout()
plt.show()

# Correlation matrix visualization
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Box plot for Calories_Burned across Experience_Level
plt.figure(figsize=(10, 6))
sns.boxplot(x='Experience_Level', y='Calories_Burned', data=data)
plt.title('Calories Burned by Experience Level')
plt.show()

# Bar chart for average BMI across Workout_Type
plt.figure(figsize=(10, 6))
sns.barplot(x='Workout_Type', y='BMI', data=data, ci=None)
plt.title('Average BMI across Workout Types')
plt.show()

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
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Save the trained model to a .pkl file
joblib.dump(regressor, 'regressor_model.pkl')

# Initialize and train the Random Forest Classifier model for experience level classification
X_classification = data[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'BMI']]
y_classification = data['Experience_Level']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_classification)

X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_encoded, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_classification, y_train_classification)

# Save the classifier model and LabelEncoder to .pkl files
joblib.dump(classifier, 'classifier_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

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
    prediction_classification = classifier.predict(user_input_classification)
    experience_level = label_encoder.inverse_transform(prediction_classification)  # Convert back to original labels
    st.write(f"Predicted Experience Level: {experience_level[0]}")
