import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load sample data
st.title("Analyzing Gym Member Exercise Data for Health & Fitness Insights")

# Tab layout for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Input Features & Prediction", "Visualization", "Correlation Heatmap"])

# Define the model as a global variable so it can be used in multiple tabs
model = LinearRegression()
gym_data = None  # Define gym_data variable outside the if-block for wider access

with tab1:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        gym_data = pd.read_csv(uploaded_file)
        st.write("Data Sample:")
        st.write(gym_data.head())
    else:
        st.warning("Please upload a CSV file to proceed.")

# Handle data preprocessing and model training if data is uploaded
if uploaded_file is not None:
    # Drop non-numeric columns or handle them appropriately
    gym_data = gym_data.select_dtypes(include=[np.number]).dropna()

    # Ensure 'Calories_Burned' is in the dataset for training
    if "Calories_Burned" in gym_data.columns:
        X = gym_data.drop(columns=["Calories_Burned"])
        y = gym_data["Calories_Burned"]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Fit the model
        model.fit(X_train, y_train)

with tab2:
    st.header("Input Features & Prediction")
    if uploaded_file is not None:
        # Create input widgets for user-defined features
        user_data = {}
        for col in X.columns:
            user_data[col] = st.number_input(f"Enter value for {col}", min_value=0.0, step=0.1)
        
        # Convert input data to DataFrame
        user_data_df = pd.DataFrame([user_data])
        
        # Display prediction if all inputs are provided
        if st.button("Predict Calories Burned"):
            try:
                # Ensure that input data matches the model's expected format
                user_data_df = user_data_df.astype(float)  # Ensure all values are numeric
                prediction = model.predict(user_data_df)
                st.success(f"Predicted Calories Burned: {prediction[0]:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload data in the 'Data Upload' tab.")

with tab3:
    st.header("Visualization")
    if uploaded_file is not None:
        # Display pairplot for initial data exploration
        st.subheader("Data Distribution by Features")
        sns.pairplot(gym_data)
        st.pyplot(plt.gcf())
    else:
        st.warning("Please upload data in the 'Data Upload' tab.")

with tab4:
    st.header("Correlation Heatmap")
    if uploaded_file is not None:
        # Plot correlation heatmap
        st.subheader("Correlation Matrix")
        plt.figure(figsize=(10, 6))
        sns.heatmap(gym_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
    else:
        st.warning("Please upload data in the 'Data Upload' tab.")
