import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set up Streamlit page
st.title('Gym Member Exercise Data Insights')

# 1. **File Upload Section**
st.subheader('Upload Your Data File')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded file into a DataFrame without headers
    gym_data = pd.read_csv(uploaded_file, header=None)

    # Assign column names manually based on provided structure
    gym_data.columns = [
        'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
        'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 
        'Workout_Type', 'Fat_Percentage', 'Water_Intake (liters)', 
        'Workout_Frequency (days/week)', 'Experience_Level', 'BMI'
    ]
    
    st.write("Data Preview:")
    st.dataframe(gym_data.head())  # Show the first few rows of the dataset

    # Convert relevant columns to numeric, coerce errors to NaN and then drop rows with NaN
    numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                    'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 
                    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'BMI']
    gym_data[numeric_cols] = gym_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    gym_data.dropna(subset=numeric_cols, inplace=True)

    # 2. **Data Visualizations**
    st.subheader('Data Visualizations')

    # Combined histogram for all numeric columns in one graph
    st.write("Combined Histogram for Numeric Columns:")
    numeric_data = gym_data[numeric_cols]
    if not numeric_data.empty:
        plt.figure(figsize=(10, 6))
        for column in numeric_data.columns:
            sns.histplot(numeric_data[column], kde=True, label=column, bins=20, edgecolor='black', kde_kws={'color': 'red'})
        plt.title('Combined Histogram of All Numeric Features')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(title='Features')
        st.pyplot()
    else:
        st.write("The dataset contains no numeric columns to plot histograms.")

    # 3. **Model Predictions: Calories Burned**
    st.subheader('Calories Burned Prediction')

    # Check if the dataset contains relevant columns (age, weight, workout type, etc.)
    required_columns = ['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Workout_Type', 'Calories_Burned']
    
    if all(col in gym_data.columns for col in required_columns):
        # Encoding Workout_Type as dummy variables
        gym_data_encoded = pd.get_dummies(gym_data, columns=['Workout_Type'], drop_first=True)
        X = gym_data_encoded[['Age', 'Weight (kg)', 'Session_Duration (hours)'] + [col for col in gym_data_encoded.columns if col.startswith('Workout_Type')]]
        y = gym_data_encoded['Calories_Burned']

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Example user input for prediction
        age = st.slider('Age', min_value=18, max_value=80, value=25)
        weight = st.number_input('Weight (kg)', min_value=40, max_value=150, value=70)
        workout_type = st.selectbox('Workout Type', options=gym_data['Workout_Type'].unique())
        session_duration = st.slider('Workout Duration (hours)', min_value=0.25, max_value=3.0, value=1.0)

        # Prepare user input for prediction
        user_data = pd.DataFrame([[age, weight, session_duration]], 
                                 columns=['Age', 'Weight (kg)', 'Session_Duration (hours)'])
        
        # Add dummy columns for Workout_Type to match model input structure
        for col in [col for col in X.columns if col.startswith('Workout_Type')]:
            user_data[col] = 1 if workout_type in col else 0

        # Make prediction
        try:
            prediction = model.predict(user_data)
            st.write(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")
        except ValueError as e:
            st.write("Error in prediction. Please ensure all inputs are valid and try again.")
    else:
        st.write("The dataset does not contain the necessary columns for prediction.")
        st.write(f"Missing columns: {set(required_columns) - set(gym_data.columns)}")
