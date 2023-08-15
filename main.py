from typing import Union
from enum import Enum

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sklearn.exceptions import InconsistentVersionWarning

from src.ml.data import feature_engineering
from src.ml.model import inference, load_model


def hyphen_to_underscore(string: str):
    """Helper function to replace hyphens with underscores."""
    return string.replace('-', '_')


class CensusData(BaseModel):
    """
    This class defines which data inputs are valid for our machine learning model.
    We provide an example input for each field.
    """

    # input_cols: list = ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status',
    #  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']

    # Automatically generate aliases for names with hyphens where underscores are used instead
    # because hyphens cannot be used in field names in Python.
    model_config = ConfigDict(alias_generator=hyphen_to_underscore)

    age: int = Field(..., example=39)
    workclass: str = Field(..., example='State-gov')
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
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

    # NOTE in pydantic v1 this is called @validator I think
    @field_validator("age")
    @classmethod
    def valid_age(cls, value):
        if value < 0:
            raise ValueError("Age cannot be negative.")

        if value > 0:
            raise ValueError("Age is higher than reasonable.")

    @field_validator("salary", "education_num", "fnlgt", "capital_gain",
                     "capital_loss", "hours_per_week")
    @classmethod
    def non_negative(cls, value):
        if value < 0:
            raise ValueError(f"Value cannot be negative. {value} received.")


app = FastAPI(
    title="Random Forest for Census Data API",
    description="API to do inference with a Random Forest.",
    version="0.0.1"
)

try:
    CLF = load_model('model/random_forest.pkl')
except InconsistentVersionWarning as warning:
    print(warning.original_sklearn_version)


@app.get("/")
async def get_root():
    return "Hello there!"


# TODO can we do post on root too?
@app.post("/inference/")
async def get_predictions(data: CensusData):

    # TODO repeat correct preprocessing and feature engineering
    # as done during training (incl. label encoder etc.)
    '''
    df = preprocessing(df)

    # Process the test data using the fitted OneHotEncoder and label binarizer
    X_test, y_test, _, _ = feature_engineering(
        df_test, categorical_features=cat_features, label=target_label,
        training=False, encoder=encoder, lb=lb
    )
    '''

    # Do some checks on input data
    if data.feature_1 < 0:
        raise HTTPException(
            status_code=400,
            detail="feature_1 needs to be above 0."
        )

    # clf: fixed? Use getter?
    # Or allow model selection?
    y = inference(CLF, data)

    return y
