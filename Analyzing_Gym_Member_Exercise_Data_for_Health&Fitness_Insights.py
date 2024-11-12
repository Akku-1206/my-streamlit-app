import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set up Streamlit page
st.title('Gym Member Exercise Data Insights')

# 1. **File Upload Section**
st.subheader('Upload Your Data File')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded file into a DataFrame
    gym_data = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.write("Data Preview:")
    st.dataframe(gym_data.head())  # Show the first few rows of the dataset

    # Display summary statistics
    st.write("Data Summary:")
    st.write(gym_data.describe())

    # 2. **Input Features for Predictions**
    st.subheader('Input Features for Prediction')

    # Example input fields for prediction
    age = st.slider('Age', min_value=18, max_value=80, value=25)
    weight = st.number_input('Weight (kg)', min_value=40, max_value=150, value=70)
    workout_type = st.selectbox('Workout Type', options=['Cardio', 'Strength', 'Flexibility'])
    session_duration = st.slider('Workout Duration (minutes)', min_value=15, max_value=180, value=45)

    # 3. **Data Visualizations**
    st.subheader('Data Visualizations')

    # Correlation Heatmap
    st.write("Correlation Heatmap:")
    if not gym_data.empty:
        corr_matrix = gym_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot()

    # 4. **Model Predictions: Calories Burned**
    st.subheader('Calories Burned Prediction')

    # Check if the dataset contains relevant columns (age, weight, workout type, etc.)
    if 'Calories_Burned' in gym_data.columns:
        # Assuming column names match the expected features in the model
        X = gym_data[['Age', 'Weight', 'Workout_Type', 'Session_Duration']]
        y = gym_data['Calories_Burned']

        # Encode categorical variables if necessary (like workout type)
        X = pd.get_dummies(X, drop_first=True)

        # Train a simple Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # User input data for prediction
        user_data = pd.DataFrame([[age, weight, workout_type, session_duration]], 
                                 columns=['Age', 'Weight', 'Workout_Type', 'Session_Duration'])

        # Encode the user's input the same way
        user_data = pd.get_dummies(user_data, drop_first=True)

        # Make prediction
        prediction = model.predict(user_data)

        # Show prediction result
        st.write(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")

        # 5. **Explanation (optional)**
        st.write("Explanation: The model uses factors such as age, weight, workout type, and session duration to predict calories burned.")
    else:
        st.write("The dataset does not contain 'Calories_Burned' column for predictions.")

else:
    st.write("Please upload a CSV file to get started.")

