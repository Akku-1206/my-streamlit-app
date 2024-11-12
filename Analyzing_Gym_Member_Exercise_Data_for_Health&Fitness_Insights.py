import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Set the title and description
st.title("Gym Members Exercise Tracking Analysis")
st.markdown("This app allows you to upload data, input details for prediction, view model performance, and explore data visualizations.")

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = None
if 'gym_data' not in st.session_state:
    st.session_state.gym_data = None

# Tab selection for navigation
tabs = ['Dataset Upload', 'Input Features', 'Prediction', 'EDA & Visualizations']
selected_tab = st.sidebar.radio("Navigation", tabs)

# Option to choose between file upload or manual input
input_choice = st.sidebar.selectbox("Choose how to provide the data", ["Upload CSV File", "Enter Data Manually"])

# 1. Dataset Upload Tab
if selected_tab == 'Dataset Upload' and input_choice == 'Upload CSV File':
    st.header("Upload Your Own Dataset")
    st.markdown("Upload a CSV file for analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Drag and drop a CSV file", type=['csv'])
    
    if uploaded_file:
        gym_data = pd.read_csv(uploaded_file)
        st.session_state.gym_data = gym_data
        st.write("Dataset loaded successfully!")
        st.dataframe(gym_data.head())  # Display the first few rows of the dataset
    else:
        st.write("Please upload a dataset to proceed.")

# 2. Input Features Tab
elif selected_tab == 'Input Features' and input_choice == 'Enter Data Manually':
    st.header('Enter Your Details for Prediction')

    # Input fields for user details
    age_input = st.slider("Age", min_value=18, max_value=100, value=25)
    weight_input = st.slider("Weight (kg)", min_value=18, max_value=200, value=70)
    session_duration = st.slider("Session Duration (hours)", min_value=0.5, max_value=3.0, value=1.0)

    # Workout type selection (one-hot encoding)
    workout_types = ['HIIT', 'Strength', 'Yoga']
    workout_input = st.selectbox("Workout Type", workout_types)
    encoded_workout = [1 if workout == workout_input else 0 for workout in workout_types]

    # Show entered details
    entered_details = {
        'Age': age_input,
        'Weight (kg)': weight_input,
        'Session_Duration (hours)': session_duration,
        'Workout_Type_HIIT': encoded_workout[0],
        'Workout_Type_Strength': encoded_workout[1],
        'Workout_Type_Yoga': encoded_workout[2]
    }

    st.write('Entered Details:')
    for key, value in entered_details.items():
        st.write(f"{key}: {value}")

    # Save input details for prediction
    st.session_state.user_input = entered_details

# 3. Prediction Tab
elif selected_tab == 'Prediction':
    st.header("Model Performance Comparison and Prediction")
    
    # Load the dataset if not already loaded
    if st.session_state.gym_data is not None:
        gym_data = st.session_state.gym_data
        
        # Ensure only numeric columns are selected for model training
        gym_data_numeric = gym_data.select_dtypes(include=[np.number])
        
        if gym_data_numeric.shape[1] == 0:
            st.write("No numeric columns found in the dataset. Please upload a valid dataset.")
        else:
            X = gym_data_numeric.drop('Calories_Burned', axis=1, errors='ignore')  # Drop the target column
            y = gym_data_numeric['Calories_Burned'] if 'Calories_Burned' in gym_data_numeric else None

            if y is not None:
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict on test set
                y_pred = model.predict(X_test)

                # Model Evaluation
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse}")
                
                # User prediction
                if st.session_state.user_input is not None:
                    user_input_df = pd.DataFrame([st.session_state.user_input])
                    user_pred = model.predict(user_input_df)
                    st.write(f"Predicted Calories Burned: {user_pred[0]} kcal")
                else:
                    st.write("Please enter details in the 'Input Features' tab to make predictions.")
            else:
                st.write("Target column 'Calories_Burned' is missing from the dataset.")
    else:
        st.write("Dataset is not loaded. Please upload a dataset first.")

# 4. Exploratory Data Analysis and Visualizations Tab
elif selected_tab == 'EDA & Visualizations':
    st.header("Exploratory Data Analysis and Visualizations")
    
    # Load the dataset if not already loaded
    if st.session_state.gym_data is not None:
        gym_data = st.session_state.gym_data

        # Ensure only numeric columns are selected for correlation analysis
        gym_data_numeric = gym_data.select_dtypes(include=[np.number])

        # Distribution of Calories Burned
        st.subheader("Distribution of Calories Burned")
        plt.figure(figsize=(8, 6))
        sns.histplot(gym_data['Calories_Burned'], kde=True)
        st.pyplot()

        # Correlation Heatmap
        if gym_data_numeric.shape[1] > 1:  # Ensure there are enough numeric columns
            st.subheader("Correlation Heatmap")
            plt.figure(figsize=(10, 8))
            sns.heatmap(gym_data_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot()

        # Pairplot of Selected Features
        st.subheader("Pairplot of Selected Features")
        selected_features = ['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Calories_Burned']
        if all(feature in gym_data.columns for feature in selected_features):
            sns.pairplot(gym_data[selected_features], hue="Workout_Type_HIIT")
            st.pyplot()
        else:
            st.write("Selected features are missing in the dataset.")
    else:
        st.write("Dataset is not loaded. Please upload a dataset first.")
