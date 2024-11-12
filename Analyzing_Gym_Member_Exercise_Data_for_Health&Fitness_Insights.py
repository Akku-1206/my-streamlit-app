import streamlit as st
import pandas as pd
import numpy as np
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

# Streamlit User Interface
st.title("Gym Members Exercise Tracking Analysis")
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Select a tab", ["Enter User Input", "Model Prediction", "Dataset Upload"])

if tabs == "Enter User Input":
    st.subheader("Enter Your Details for Prediction")
    age = st.slider('Age', 18, 100, 25)
    weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
    session_duration = st.slider('Session Duration (hours)', 0.5, 3.0, 1.0)
    workout_type = st.selectbox('Workout Type', ['HIIT', 'Strength', 'Yoga'])

    # User input for prediction
    user_input = {
        'Age': age,
        'Weight (kg)': weight,
        'Session_Duration (hours)': session_duration,
        'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
        'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
        'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0,
    }
    user_input_df = pd.DataFrame(user_input, index=[0])

    st.write(f"Entered Details: {user_input}")

elif tabs == "Model Prediction":
    st.subheader("Model Performance Comparison")

    # Model training and evaluation
    if 'gym' not in locals():
        st.error("Dataset is not loaded. Please upload a dataset first.")
    else:
        X = gym.drop('Calories_Burned', axis=1)
        y = gym['Calories_Burned']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

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
        st.write(results_df)

        # Select model for prediction
        selected_model_name = st.selectbox("Select Model for Prediction", ["Linear Regression", "Decision Tree", "Random Forest"])
        selected_model = models[selected_model_name]

        # Make prediction for user input
        if user_input_df.empty:
            st.error("Please enter user input first.")
        else:
            user_prediction = selected_model.predict(user_input_df)[0]
            st.subheader(f"Predicted Calories Burned: {user_prediction:.2f} kcal")

            # Explanation of factors contributing to the prediction
            st.write("### Factors Contributing to the Prediction")
            st.write(f"Age: {age}")
            st.write(f"Weight: {weight} kg")
            st.write(f"Session Duration: {session_duration} hours")
            st.write(f"Workout Type: {workout_type}")

elif tabs == "Dataset Upload":
    st.subheader("Upload Your Own Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

    if uploaded_file is not None:
        gym = pd.read_csv(uploaded_file)
        gym = preprocess_data(gym)
        st.write(gym.head())
