"""
main.py

Project Step 4: API
A RESTful API built with FastAPI.

Rubric (API Creation):
- The API must implement GET and POST.
- GET on the root domain gives a greeting.
- POST on a different path does model inference.
- POST result should be either ">50K" or "<=50K".

Run with:
    > uvicorn main:app --reload
"""

import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# Categorical features used during training.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class Data(BaseModel):
    """Pydantic model describing a single inference request.

    Field aliases use hyphens to match the census column names.
    """

    # TODO (Step 4): Define all input fields with example values and aliases
    # that match the dataset columns.
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# Load the model and encoder once at startup.
project_path = os.path.dirname(os.path.abspath(__file__))
encoder = load_model(os.path.join(project_path, "model", "encoder.pkl"))
model = load_model(os.path.join(project_path, "model", "model.pkl"))

app = FastAPI()


@app.get("/")
async def get_root():
    """GET on the root domain returns a welcome/greeting message."""
    # TODO (Step 4): Return a greeting message.
    return {"message": "Hello from the API!"}


@app.post("/data/")
async def post_inference(data: Data):
    """POST that runs model inference and returns the predicted salary class."""
    # TODO (Step 4): Convert the request into a DataFrame, process it with
    # process_data (training=False, reusing encoder), run inference, and
    # return the labeled result.
    data_dict = data.dict(by_alias=True)
    data_df = pd.DataFrame.from_dict(
        {k: [v] for k, v in data_dict.items()}
    )

    X, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
    )
    preds = inference(model, X)
    return {"result": apply_label(preds)}
