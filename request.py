import requests
r = requests.post(
    "http://localhost:8000/api/predict",
    json={"input": "свечи"}
)
print(r.json())