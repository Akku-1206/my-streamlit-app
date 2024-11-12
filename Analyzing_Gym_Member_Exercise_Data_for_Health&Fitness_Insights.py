import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load your data (replace this line with your actual file path or data loading logic)
# Example: df = pd.read_csv('your_dataset.csv')
df = pd.DataFrame()  # Placeholder for your actual dataset

# Check if dataset is empty or not loaded
if df.empty:
    raise ValueError("The dataset is empty. Please load your data.")

# Step 2: Feature Engineering (BMI calculation if not present)
if 'BMI' not in df.columns:
    if 'Weight (kg)' in df.columns and 'Height (m)' in df.columns:
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
    else:
        raise ValueError("Required columns 'Weight (kg)' or 'Height (m)' are missing.")

# Step 3: Exploratory Data Analysis (EDA)

# Histograms for numeric columns
df.hist(bins=10, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Columns")
plt.show()

# Boxplots for numeric columns to check for outliers
numeric_columns = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Step 4: Prepare Data for Machine Learning
# Let's say we want to predict "Calories_Burned" based on other features.
X = df[['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM', 
        'Session_Duration (hours)', 'Fat_Percentage', 'Water_Intake (liters)', 
        'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']]
y = df['Calories_Burned']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Coefficients of the model
print(f"Model Coefficients:\n{model.coef_}")

# Step 7: Visualize Predictions vs Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Actual vs Predicted Calories Burned")
plt.show()
