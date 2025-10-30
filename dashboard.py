import streamlit as st
import requests

st.title("🏥 Patient Risk Prediction System")
st.write("Enter patient details below:")

age = st.number_input("Age", min_value=0, max_value=120, value=60)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=85)
glucose_level = st.number_input("Glucose Level", min_value=50, max_value=400, value=130)
previous_admissions = st.number_input("Previous Admissions", min_value=0, max_value=10, value=1)

if st.button("Predict Risk"):
    data = {
        "age": age,
        "heart_rate": heart_rate,
        "glucose_level": glucose_level,
        "previous_admissions": previous_admissions
    }
    
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['risk_label']}")
            st.write(f"Confidence Score: {result['confidence_score']}")
        else:
            st.error(f"Server Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
