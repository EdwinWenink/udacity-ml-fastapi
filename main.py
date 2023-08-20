"""
FastAPI using Pydantic v1
"""

from typing import Union
from enum import Enum

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, validator, ConfigDict
from sklearn.exceptions import InconsistentVersionWarning
import pandas as pd

from src.ml.data import preprocessing, feature_engineering
from src.ml.model import inference, load_model


def underscore_to_hyphen(string: str):
    """Helper function to replace hyphens with underscores."""
    return string.replace('_', '-')


class CensusData(BaseModel):
    """
    This class defines which data inputs are valid for our machine learning model.
    We provide an example input for each field.
    """
    # Automatically generate aliases for names with hyphens where underscores are used instead
    # because hyphens cannot be used in field names in Python. In Pydantic v2 this would be:
    # model_config = ConfigDict(alias_generator=hyphen_to_underscore)
    # In v1 (1.10) this is written as:
    class Config:
        alias_generator = underscore_to_hyphen

    age: int = Field(..., example=39)
    workclass: str = Field(..., example='State-gov')
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)  # alias
    marital_status: str = Field(..., example="Never-married")  # alias
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174)  # alias
    capital_loss: int = Field(..., example=0)  # alias
    hours_per_week: int = Field(..., example=40)  # alias
    native_country: str = Field(..., example="United-States")  # alias
    salary: str = Field(..., example="<=50K")

    # NOTE in pydantic v2 this is called @field_validator
    @validator("age")
    @classmethod
    def valid_age(cls, value):
        if value < 0:
            raise ValueError("Age cannot be negative.")

        if value > 130:
            raise ValueError("Age is higher than reasonable.")

        return value

    @validator("education_num", "fnlgt", "capital_gain",
               "capital_loss", "hours_per_week")
    @classmethod
    def non_negative(cls, value):
        if value < 0:
            raise ValueError(f"Value cannot be negative. {value} received.")
        return value


app = FastAPI(
    title="Random Forest for Census Data API",
    description="API to do inference with a Random Forest.",
    version="0.0.1"
)

try:
    # TODO use DVC; this model is not yet available in the repo
    # NOTE this object should include required data transforms
    INFERENCE_ARTIFACT = load_model('model/random_forest.pkl')
except InconsistentVersionWarning as warning:
    print(warning.original_sklearn_version)


@app.get("/")
async def get_root():
    return "Hello there!"


@app.post("/inference/")
async def get_prediction(data: CensusData):

    # NOTE we have to use the alias here!
    # In Pydantic v2, use model_dump(by_alias=True)
    d = data.dict(by_alias=True)

    # For a single data point we need to manually pass an index to construct a dataframe
    df = pd.DataFrame.from_records(d, index=[0])

    # Repeat basic data cleaning
    df = preprocessing(df)

    # Perform inference. The inference artifact includes data transforms.
    y = inference(INFERENCE_ARTIFACT, df)
    return {'prediction': int(y)}
