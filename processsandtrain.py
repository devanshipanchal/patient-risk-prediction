# preprocess_and_train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Data Simulation and Collection ---
# In a real scenario, you would load data from a database or CSV file.
# Example: df = pd.read_csv('patient_data.csv')
data = {
    'age': [65, 72, 45, 55, 78, 81, np.nan, 62, 59, 88],
    'heart_rate': [85, 92, 76, 88, 95, 102, 78, 81, np.nan, 99],
    'glucose_level': [120, 140, 110, 130, 160, 180, 115, 125, 145, np.nan],
    'previous_admissions': [1, 2, 0, 1, 3, 2, 0, 1, 2, 4],
    'readmitted': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1] # Target variable: 1 for Yes, 0 for No
}
df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

# --- 2. Preprocessing ---
# Separate features (X) and target (y)
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Handle missing values using SimpleImputer (replaces NaN with the mean)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Save the imputer and scaler to use them for new data in the API
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("\n--- Data Preprocessed and Scaled ---")

# --- 3. Model Training (Random Forest) ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# --- 5. Save the Trained Model ---
# The model is saved to a file for later use by the API
joblib.dump(model, 'patient_risk_model.joblib')
print("\nModel saved to 'patient_risk_model.joblib'")

