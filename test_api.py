import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "age": 60,
    "heart_rate": 85,
    "glucose_level": 130,
    "previous_admissions": 1
}

response = requests.post(url, json=data)
print(response.json())
