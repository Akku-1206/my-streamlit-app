import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Streamlit App Title
st.title("Analyzing Gym Member Exercise Data for Health & Fitness Insights")

# Cache functions for loading data and model to optimize memory usage
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    print(data.columns)  # Print the column names
    return data.sample(frac=0.1, random_state=1)  # Use 10% of the data for optimization

@st.cache_resource
def train_model(data):
    X = data[['age', 'weight', 'session_duration']]
    y = data['calories_burned']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Sidebar for file upload
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Tabs for App Sections
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Input Features & Prediction", "Visualization", "Correlation Heatmap"])

# Tab 1: Data Upload
with tab1:
    st.header("Data Upload")
    if uploaded_file:
        gym_data = load_data(uploaded_file)
        st.write("Data Sample:")
        st.write(gym_data.head())
    else:
        st.write("Please upload a CSV file.")

# Tab 2: Input Features & Prediction
with tab2:
    st.header("Input Features & Prediction")
    if uploaded_file:
        # Get input features from the user
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, step=0.1)
        session_duration = st.number_input("Session Duration (minutes)", min_value=0, max_value=300, step=1)
        
        # Prepare input data for prediction
        user_data = pd.DataFrame({
            'age': [age],
            'weight': [weight],
            'session_duration': [session_duration]
        })
        
        # Train and make predictions
        model = train_model(gym_data)
        prediction = model.predict(user_data[['age', 'weight', 'session_duration']])
        st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")
    else:
        st.write("Please upload data in the 'Data Upload' tab.")

# Tab 3: Visualization
with tab3:
    st.header("Visualization")
    if uploaded_file:
        st.subheader("Data Distribution by Features")
        selected_columns = st.multiselect("Select columns for distribution plot", gym_data.columns.tolist())
        if selected_columns:
            for col in selected_columns:
                fig, ax = plt.subplots()
                sns.histplot(gym_data[col], ax=ax)
                st.pyplot(fig)
    else:
        st.write("Please upload data in the 'Data Upload' tab.")

# Tab 4: Correlation Heatmap
with tab4:
    st.header("Correlation Heatmap")
    if uploaded_file:
        # Get a list of numeric columns
        numeric_cols = gym_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Display a multiselect for choosing columns for the heatmap
        selected_cols = st.multiselect("Select columns for heatmap", numeric_cols, default=numeric_cols[:5])
        
        if selected_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(gym_data[selected_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
    else:
        st.write("Please upload data in the 'Data Upload' tab.")
