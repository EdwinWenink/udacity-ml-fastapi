"""
FastAPI provides a TestClient so that you can do local testing with pytest.
It behaves like the well known requests module. For testing of a live API
use the requests module to make API calls.
"""

import json
from typing import Union

from fastapi.testclient import TestClient
from pydantic import BaseModel

from main import app

client = TestClient(app)

VALID_INFERENCE_INPUT = {
        "age": 39,
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


# Notice the negative age
INVALID_INFERENCE_INPUT = {
        "age": -1,
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


def test_greeting_at_root():
    r = client.get("/")
    assert r.status_code == 200


def test_post_inference_valid_input():
    r = client.post("/inference/", data=json.dumps(VALID_INFERENCE_INPUT))
    assert r.status_code == 200


def test_post_inference_output_type():
    r = client.post("/inference/", data=json.dumps(VALID_INFERENCE_INPUT))
    r_json = r.json()
    assert isinstance(r_json['prediction'], int)


def test_post_inference_invalid_input():
    r = client.post("/inference/", data=json.dumps(INVALID_INFERENCE_INPUT))
    print(r.status_code)
    # Validation Error response
    assert r.status_code == 422
