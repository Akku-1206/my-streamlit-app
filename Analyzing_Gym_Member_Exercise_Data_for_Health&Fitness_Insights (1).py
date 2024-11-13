#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the essential libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Load the dataset
file_path = r"C:\Users\ASUS\OneDrive\Desktop\gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

# Display basic info and the first few rows of the dataset
data_info = data.info()
data_head = data.head()

data_info, data_head




# In[3]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate features into categorical and numerical
categorical_features = ['Gender', 'Workout_Type']
numerical_features = [col for col in data.columns if col not in categorical_features]

# Define Column Transformer for encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Apply transformations to the dataset
processed_data = preprocessor.fit_transform(data)

# Create a DataFrame to view the processed data
processed_data_df = pd.DataFrame(processed_data, columns=numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

processed_data_df.head()



# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure for distributions of key metrics
plt.figure(figsize=(18, 5))

# Plot distribution for Calories_Burned
plt.subplot(1, 3, 1)
sns.histplot(processed_data_df['Calories_Burned'], kde=True, color='blue')
plt.title('Distribution of Calories Burned')

# Plot distribution for BMI
plt.subplot(1, 3, 2)
sns.histplot(processed_data_df['BMI'], kde=True, color='green')
plt.title('Distribution of BMI')

# Plot distribution for Fat_Percentage
plt.subplot(1, 3, 3)
sns.histplot(processed_data_df['Fat_Percentage'], kde=True, color='purple')
plt.title('Distribution of Fat Percentage')

plt.tight_layout()
plt.show()



# In[5]:


# Select only the numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[6]:


# Box plot for Calories_Burned across Experience_Level
plt.figure(figsize=(10, 6))
sns.boxplot(x='Experience_Level', y='Calories_Burned', data=data)
plt.title('Calories Burned by Experience Level')
plt.show()

# Bar chart for average BMI across Workout_Type
plt.figure(figsize=(10, 6))
sns.barplot(x='Workout_Type', y='BMI', data=data, ci=None)
plt.title('Average BMI across Workout Types')
plt.show()


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Define features and target for regression
X = processed_data_df.drop(columns=['Calories_Burned'])
y = processed_data_df['Calories_Burned']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')


# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Define features and target variable
X = processed_data_df.drop(columns=['Calories_Burned'])
y = processed_data_df['Calories_Burned']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_model.predict(X_test)

# Evaluate the model
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Regression Results:")
print(f"Mean Absolute Error: {mae_linear}")
print(f"R-squared: {r2_linear}\n")




# In[9]:


# Initialize and train the Decision Tree Regressor model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree_model.predict(X_test)

# Evaluate the model
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("Decision Tree Regressor Results:")
print(f"Mean Absolute Error: {mae_tree}")
print(f"R-squared: {r2_tree}\n")


# In[10]:


# Initialize and train the Random Forest Regressor model
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)

# Make predictions
y_pred_forest = forest_model.predict(X_test)

# Evaluate the model
mae_forest = mean_absolute_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print("Random Forest Regressor Results:")
print(f"Mean Absolute Error: {mae_forest}")
print(f"R-squared: {r2_forest}\n")


# In[12]:


get_ipython().system('pip install streamlit')


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\gym_members_exercise_tracking.csv")

# Assume columns like Age, Weight, Session_Duration, Calories_Burned exist (this should be based on your dataset)
# You need to adjust these column names based on the actual dataset
X = data[['Age', 'Weight (kg)', 'Session_Duration (hours)']]  # Features (replace with relevant columns)
y = data['Calories_Burned']  # Target variable (replace with your target column)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but useful for models like linear regression or neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model (you can use other models as well, such as RandomForestRegressor)
regressor = LinearRegression()

# Train the model
regressor.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(regressor, 'regressor_model.pkl')


# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\gym_members_exercise_tracking.csv")

# Assume 'Experience_Level' is the target variable, and other columns are features
X = data[['Age', 'Weight (kg)', 'Session_Duration (hours)', 'BMI']]  # Use relevant features (adjust column names)
y = data['Experience_Level']  # The target variable (categorical)

# Encode categorical target variable if it's in string format (e.g., 'Beginner', 'Intermediate', 'Expert')
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the classifier (RandomForestClassifier in this case)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(classifier, 'classifier_model.pkl')

# Save the LabelEncoder to convert predictions back to original labels
joblib.dump(label_encoder, 'label_encoder.pkl')


# In[19]:


import streamlit as st
import pandas as pd
import joblib

# Load models
regressor = joblib.load('regressor_model.pkl')
classifier = joblib.load('classifier_model.pkl')

st.title("Health & Fitness Tracker")

# Data upload
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", input_data.head())

# Input sliders
age = st.slider("Age", 18, 80, 25)
weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0)
session_duration = st.slider("Session Duration (hours)", 0.0, 5.0, 1.0)

# Display Predictions
if st.button("Predict Calories Burned"):
    user_input = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight', 'Session_Duration'])
    prediction = regressor.predict(user_input)
    st.write(f"Predicted Calories Burned: {prediction[0]:.2f}")

if st.button("Classify Experience Level"):
    user_input = pd.DataFrame([[age, weight, session_duration]], columns=['Age', 'Weight', 'Session_Duration'])
    prediction = classifier.predict(user_input)
    st.write(f"Predicted Experience Level: {prediction[0]}")


# In[ ]:




