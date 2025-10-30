# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Saved Model and Preprocessing Objects ---
try:
    model = joblib.load('patient_risk_model.joblib')
    imputer = joblib.load('imputer.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    model = None
    imputer = None
    scaler = None
    print("Error: Model or preprocessing files not found. Please run 'preprocess_and_train.py' first.")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives patient data in JSON format, preprocesses it,
    and returns a readmission risk prediction.
    """
    if not all([model, imputer, scaler]):
        return jsonify({"error": "Model not loaded. Please ensure training is complete."}), 500

    # Get data from the POST request
    data = request.get_json(force=True)
    
    # --- Data Validation ---
    required_features = ['age', 'heart_rate', 'glucose_level', 'previous_admissions']
    if not all(feature in data for feature in required_features):
        return jsonify({"error": "Missing data. Please provide all features: " + ", ".join(required_features)}), 400
    
    # Create a DataFrame from the input data to maintain feature order
    try:
        input_df = pd.DataFrame([data])
        features_ordered = input_df[required_features]
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input format or missing features: {e}"}), 400

    # --- Preprocess the input data using the saved imputer and scaler ---
    features_imputed = imputer.transform(features_ordered)
    features_scaled = scaler.transform(features_imputed)
    
    # --- Make a prediction ---
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    
    # --- Return the response in JSON format ---
    # Convert numpy int64 to a native Python int for JSON serialization
    output = int(prediction[0])
    risk_probability = float(probability[0][1]) # Probability of the '1' class (High Risk)
    
    response = {
        'prediction': output,
        'risk_label': 'High Risk' if output == 1 else 'Low Risk',
        'confidence_score': f"{risk_probability:.2f}"
    }
    
    return jsonify(response)

@app.route('/')
def home():
    """A simple welcome message to verify the server is running."""
    return "Patient Risk Prediction API is running. Use the /predict endpoint to get a prediction."

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
