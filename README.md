# 🩺 Patient Risk Prediction System

A Machine Learning–powered Flask API and Streamlit app that predicts patient readmission risk based on health data such as age, heart rate, glucose level, and previous admissions.

---

## 🚀 Project Overview

This project aims to assist healthcare providers in identifying patients at high risk of hospital readmission using data-driven insights.  
It includes:

- A **Flask API** that serves the ML model for predictions.  
- A **Streamlit UI** that provides an interactive dashboard for doctors or analysts.  
- Preprocessing and training scripts for reproducibility.

---

## 🧠 Tech Stack

| Component | Technology Used |
|------------|----------------|
| Backend | Flask |
| Frontend | Streamlit |
| Machine Learning | scikit-learn, pandas, numpy |
| Model Persistence | joblib |
| Deployment | Git + GitHub |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/DevanshiPanchal/patient-risk-prediction.git
cd patient-risk-prediction

2️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate  # For Linux/Mac

3️⃣ Install dependencies
pip install -r requirements.txt

🧩 Run the Flask API
python app.py

You can test it by sending a POST request to /predict with JSON input:

{
  "age": 45,
  "heart_rate": 88,
  "glucose_level": 130,
  "previous_admissions": 2
}

🎨 Run the Streamlit App
streamlit run ui.py


Then open your browser at:

http://localhost:8501

🧾 API Endpoints
Endpoint	Method	Description
/	GET	Health check endpoint
/predict	POST	Returns risk prediction and confidence score

🧠 Model Training

Run the preprocessing and training script before starting the app:

python preprocess_and_train.py


This will generate:

patient_risk_model.joblib
imputer.joblib
scaler.joblib
