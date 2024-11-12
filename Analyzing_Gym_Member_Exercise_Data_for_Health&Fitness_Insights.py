# User Input for Prediction
st.subheader("Enter Your Details for Prediction")
age = st.slider('Age', 18, 100, 25)
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
session_duration = st.slider('Session Duration (hours)', 0.5, 3.0, 1.0)
workout_type = st.selectbox('Workout Type', ['HIIT', 'Strength', 'Yoga'])

# Data Preprocessing for User Input
user_input = {
    'Age': age,
    'Weight (kg)': weight,
    'Session_Duration (hours)': session_duration,
    'Experience_Level': 0,  # Assuming Experience_Level is encoded. Adjust based on requirements.
    'Gender_Male': 0,  # Assuming gender column exists; add more based on one-hot encoding.
    'Workout_Type_HIIT': 1 if workout_type == 'HIIT' else 0,
    'Workout_Type_Strength': 1 if workout_type == 'Strength' else 0,
    'Workout_Type_Yoga': 1 if workout_type == 'Yoga' else 0,
    # Add other missing columns with default values (like zeros) if needed
}

# Convert user input into a DataFrame
user_input_df = pd.DataFrame(user_input, index=[0])

# Align user_input_df with the columns of X_train
user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)

# Model Training
X_train, X_test, y_train, y_test = train_test_split(gym.drop('Calories_Burned', axis=1), gym['Calories_Burned'], test_size=0.2, random_state=42)

# Select model for prediction
selected_model_name = st.selectbox("Select Model for Prediction", ["Linear Regression", "Decision Tree", "Random Forest"])
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}
selected_model = models[selected_model_name]

# Train the selected model and make a prediction for user input
selected_model.fit(X_train, y_train)
user_prediction = selected_model.predict(user_input_df)[0]
st.subheader(f"Predicted Calories Burned: {user_prediction:.2f} kcal")

# Explanation of factors contributing to the prediction
st.subheader("Factors Contributing to the Prediction")
st.write(f"Age: {age}")
st.write(f"Weight: {weight} kg")
st.write(f"Session Duration: {session_duration} hours")
st.write(f"Workout Type: {workout_type}")
