import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Initialize Streamlit app
st.title("Analyzing Gym Member Exercise Data for Health & Fitness Insights")

# Set up tabs for each section
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Input Features & Prediction", "Visualization", "Correlation Heatmap"])

# Define global variables
model = LinearRegression()
gym_data = None  # Data variable accessible across tabs

# Tab 1: Data Upload
with tab1:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        gym_data = pd.read_csv(uploaded_file)
        st.write("Data Sample:")
        st.write(gym_data.head())
    else:
        st.warning("Please upload a CSV file to proceed.")

# Proceed with data preprocessing and model training if data is uploaded
if uploaded_file is not None:
    # Keep only numeric data and drop rows with missing values
    gym_data = gym_data.select_dtypes(include=[np.number]).dropna()

    # Ensure 'Calories_Burned' exists for prediction
    if "Calories_Burned" in gym_data.columns:
        X = gym_data.drop(columns=["Calories_Burned"])
        y = gym_data["Calories_Burned"]

        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model.fit(X_train, y_train)

# Tab 2: Input Features & Prediction
with tab2:
    st.header("Input Features & Prediction")
    if uploaded_file is not None:
        # Capture user inputs for each feature in X
        user_data = {}
        for col in X.columns:
            user_data[col] = st.number_input(f"Enter value for {col}", min_value=0.0, step=0.1)
        
        # Convert input to DataFrame format
        user_data_df = pd.DataFrame([user_data])
        
        if st.button("Predict Calories Burned"):
            try:
                # Ensure all inputs are numeric
                user_data_df = user_data_df.astype(float)
                prediction = model.predict(user_data_df)
                st.success(f"Predicted Calories Burned: {prediction[0]:.2f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload data in the 'Data Upload' tab.")

# Tab 3: Visualization
with tab3:
    st.header("Visualization")
    if uploaded_file is not None:
        st.subheader("Data Distribution by Features")
        
        # Create and display the pairplot as a static image
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.pairplot(gym_data)
        st.pyplot(fig)  # Display the figure in Streamlit
    else:
        st.warning("Please upload data in the 'Data Upload' tab.")

# Tab 4: Correlation Heatmap
with tab4:
    st.header("Correlation Heatmap")
    if uploaded_file is not None:
        st.subheader("Correlation Matrix")
        
        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(gym_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)  # Display the figure in Streamlit
    else:
        st.warning("Please upload data in the 'Data Upload' tab.")
