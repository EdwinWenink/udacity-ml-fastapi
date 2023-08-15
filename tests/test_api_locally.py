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


def test_greeting_at_root():
    r = client.get("/")
    assert r.status_code == 200
