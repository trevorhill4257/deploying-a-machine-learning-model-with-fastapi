"""
local_api.py

Project Step 4: API interaction
Uses the requests module to do one GET request and one POST request against
the locally running API.

Rubric (API Creation - API interaction):
- The script has both GET and POST requests.
- Tested locally by running this script while the API is running.
- A screenshot local_api.png shows the successful status codes and results.

Usage:
    1. In one terminal: uvicorn main:app --reload
    2. In another terminal: python local_api.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

# TODO (Step 4): Send a GET request using the URL and print the status code
# and the welcome message.
r = requests.get(f"{BASE_URL}/")
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()['message']}")

# Sample input for the POST request.
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO (Step 4): Send a POST request using the data above and print the
# status code and the result.
r = requests.post(f"{BASE_URL}/data/", json=data)
print(f"Status Code: {r.status_code}")
print(f"Result: {r.json()['result']}")
