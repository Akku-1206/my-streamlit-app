import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# App Title
st.title('Gym Members Exercise Tracking Analysis')

# Tabs
tabs = ['Dataset Upload', 'Input Features', 'Model Prediction', 'EDA & Visualizations']
selected_tab = st.selectbox('Navigation', tabs)

# Dataset Upload Tab
if selected_tab == 'Dataset Upload':
    st.header('Upload Your Own Dataset')
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

    if uploaded_file is not None:
        gym_data = pd.read_csv(uploaded_file)
        st.write(gym_data.head())  # Display first few rows of dataset
        st.session_state.gym_data = gym_data  # Save the dataset in session state

# Input Features Tab
elif selected_tab == 'Input Features':
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

    st.write('Entered Details:', entered_details)

    # Save input details for prediction
    st.session_state.user_input = entered_details

# Model Prediction Tab
elif selected_tab == 'Model Prediction':
    st.header('Model Performance Comparison and Prediction')

    if 'gym_data' not in st.session_state or 'user_input' not in st.session_state:
        st.warning("Please upload a dataset and enter your details for prediction.")
    else:
        gym_data = st.session_state.gym_data
        user_input = st.session_state.user_input

        # Ensure dataset has the required column 'Calories_Burned'
        if 'Calories_Burned' not in gym_data.columns:
            st.warning("The dataset must contain the 'Calories_Burned' column.")
        else:
            # Split the data into features (X) and target (y)
            X = gym_data.drop('Calories_Burned', axis=1)
            y = gym_data['Calories_Burned']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train the model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Model Mean Squared Error: {mse}")

            # Make prediction for user input
            user_input_df = pd.DataFrame([user_input])  # Convert input to DataFrame
            user_input_scaled = scaler.transform(user_input_df)  # Scale the input data
            user_prediction = model.predict(user_input_scaled)

            st.write(f"Predicted Calories Burned: {user_prediction[0]}")

# EDA & Visualizations Tab
elif selected_tab == 'EDA & Visualizations':
    st.header('Exploratory Data Analysis and Visualizations')

    if 'gym_data' not in st.session_state:
        st.warning("Please upload a dataset to view visualizations.")
    else:
        gym_data = st.session_state.gym_data

        # Display distribution of calories burned
        st.subheader('Distribution of Calories Burned')
        sns.histplot(gym_data['Calories_Burned'], kde=True)
        st.pyplot()

        # Correlation heatmap
        st.subheader('Correlation Heatmap')
        correlation_matrix = gym_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        st.pyplot()

        # Pairplot of selected features
        selected_features = ['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Calories_Burned']
        st.subheader('Pairplot of Selected Features')
        sns.pairplot(gym_data[selected_features], hue="Workout_Type_HIIT")
        st.pyplot()
