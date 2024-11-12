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
    
    st.write("Data Preview:")
    st.dataframe(gym_data.head())  # Show the first few rows of the dataset

    # Print the column names to check for any discrepancies
    st.write("Column Names in Dataset:")
    st.write(gym_data.columns)

    # Display summary statistics
    st.write("Data Summary:")
    st.write(gym_data.describe())

    # 2. **Input Features for Predictions**
    st.subheader('Input Features for Prediction')

    # Example input fields for prediction
    age = st.slider('Age', min_value=18, max_value=80, value=25)
    weight = st.number_input('Weight (kg)', min_value=40, max_value=150, value=70)
    workout_type = st.selectbox('Workout Type', options=['Cardio', 'Strength', 'Flexibility'])
    session_duration = st.slider('Workout Duration (hours)', min_value=0.25, max_value=3.0, value=1.0)  # in hours

    # 3. **Data Visualizations**
    st.subheader('Data Visualizations')

    # Correlation Heatmap
    st.write("Correlation Heatmap:")
    if uploaded_file:
        # Filter only numeric columns before calculating correlation
        numeric_data = gym_data.select_dtypes(include=['number'])
        
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot()
        else:
            st.write("The dataset contains no numeric columns to compute correlation.")

    # 4. **Model Predictions: Calories Burned**
    st.subheader('Calories Burned Prediction')

    # Check if the dataset contains relevant columns (age, weight, workout type, etc.)
    required_columns = ['Age', 'Weight (kg)', 'Workout_Type', 'Session_Duration (hours)', 'Calories_Burned']
    
    if all(col in gym_data.columns for col in required_columns):
        # Prepare data for training the model
        X = gym_data[['Age', 'Weight (kg)', 'Workout_Type', 'Session_Duration (hours)']]
        y = gym_data['Calories_Burned']

        # Encode categorical variables if necessary (like workout type)
        X = pd.get_dummies(X, drop_first=True)

        # Train a simple Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # User input data for prediction
        user_data = pd.DataFrame([[age, weight, workout_type, session_duration]], 
                                 columns=['Age', 'Weight (kg)', 'Workout_Type', 'Session_Duration (hours)'])

        # Encode the user's input the same way
        user_data = pd.get_dummies(user_data, drop_first=True)

        # Make prediction
        prediction = model.predict(user_data)

        # Display the predicted calories burned in a human-readable format
        st.write(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")

        # 5. **Explanation (optional)**
        st.write("Explanation: The model uses factors such as age, weight, workout type, and session duration to predict calories burned.")
    else:
        st.write("The dataset does not contain the necessary columns for prediction.")
        st.write(f"Missing columns: {set(required_columns) - set(gym_data.columns)}")

# Run the Streamlit app
