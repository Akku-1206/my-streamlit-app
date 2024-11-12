import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# App title and description
st.title("Gym Member Exercise Data Analysis")
st.write("Analyze data and predict calories burned based on user inputs.")

# Initialize model options
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Use tabs for different sections
tab1, tab2, tab3 = st.tabs(["Upload Data", "Enter User Input", "Predict Calories Burned"])

# Tab 1: Upload Data
with tab1:
    st.header("Upload Gym Data")
    uploaded_file = st.file_uploader("Upload your gym data CSV file", type="csv")
    
    if uploaded_file is not None:
        gym = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully.")
        st.write(gym.head())  # Display the first few rows of the dataset
        
        # Define target variable and features
        X = gym.drop('Calories_Burned', axis=1)
        y = gym['Calories_Burned']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.session_state['X_train'] = X_train  # Store in session state for access in other tabs
        st.session_state['y_train'] = y_train
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Tab 2: User Input Form
with tab2:
    st.header("Enter Your Details for Prediction")
    
    age = st.slider('Age', 18, 100, 25)
    weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
    session_duration = st.slider('Session Duration (hours)', 0.5, 3.0, 1.0)
    workout_type = st.selectbox('Workout Type', ['HIIT', 'Strength', 'Yoga'])
    
    # Process user input into DataFrame
    user_input = {
        'Age': age,
        'Weight (kg)': weight,
        'Session_Duration (hours)': session_duration,
        'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
        'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
        'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0,
    }
    user_input_df = pd.DataFrame(user_input, index=[0])
    
    if 'X_train' in st.session_state:
        # Match user input to X_train columns
        user_input_df = user_input_df.reindex(columns=st.session_state['X_train'].columns, fill_value=0)
        
        st.session_state['user_input_df'] = user_input_df  # Store in session state for prediction access
        st.write("User input saved successfully.")
    else:
        st.warning("Please upload and process the dataset in the first tab.")

# Tab 3: Prediction
with tab3:
    st.header("Predict Calories Burned")
    
    # Select Model
    selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
    selected_model = models[selected_model_name]
    
    if 'X_train' in st.session_state and 'user_input_df' in st.session_state:
        # Train the selected model
        selected_model.fit(st.session_state['X_train'], st.session_state['y_train'])
        
        # Make prediction with user input
        user_prediction = selected_model.predict(st.session_state['user_input_df'])[0]
        
        # Display prediction result
        st.subheader("Model Prediction")
        st.write(f"Predicted Calories Burned: {user_prediction:.2f} kcal")
    else:
        st.warning("Please complete data upload and user input in the previous tabs.")
