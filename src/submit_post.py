import requests

# This is hardcoded. After project submission this API will not be online
# and this script will fail.
API_base_endpoint = 'https://model-inference-using-fastapi.onrender.com/'
API_inference_endpoint = f"{API_base_endpoint}inference/"

# Data for inference
json_data = {"age": 39,
             "workclass": "State-gov",
             "fnlgt": 77516,
             "education": "Bachelors",
             "education-num": 13,
             "marital-status": "Never-married",
             "occupation": "Adm-clerical",
             "relationship": "Not-in-family",
             "race": "White",
             "sex": "Male",
             "capital-gain": 2174,
             "capital-loss": 0,
             "hours-per-week": 40,
             "native-country": "United-States",
             "salary": "<=50K"
             }

# Submit POST request
print("Submitting POST to", API_inference_endpoint)
print("Post JSON body:", json_data)
r = requests.post(url=API_inference_endpoint, json=json_data)

# Print response
print("Status code:", r.status_code)
print("Response:", r.json())
