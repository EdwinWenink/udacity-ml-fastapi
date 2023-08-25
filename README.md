# Deploying a machine learning model using FastAPI

[![pytest](https://github.com/EdwinWenink/udacity-ml-fastapi/actions/workflows/pytest.yml/badge.svg)](https://github.com/EdwinWenink/udacity-ml-fastapi/actions/workflows/pytest.yml)

This is a project deploying a very simple machine learning model without bells and whistles using FastAPI.
The ML code is simplistic on purpose, because the focus of this demo project is on inference using an API, testing, and automated workflows.

## Input data

This project uses a cleaned version of the [Census Income](https://archive.ics.uci.edu/dataset/20/census+income) data set.
The goal is to predict whether a person's income exceeds $50K per year based on census data (i.e. a binary classification problem).

## Install dependencies

Install the project dependencies with `pip install -r requirements.txt`.
Python 3.8.17 was used during development.

## Run API locally

To run FastAPI locally, run:

```
uvicorn main:app --reload
```

This will deploy the API at a local address such as `http://127.0.0.1:8000`.
By opening this address at the root you will be greeted with a welcome message.

To see the API docs, visit the API address appended with `/docs` in your browser, such as: `http://127.0.0.1:8000/docs`.

## Run tests

Unit tests, including local tests of the API, can be run with `python -m pytest tests`.

To test the live API, see `src/submit_post.py` for an example on how to submit a POST request for model inference.

## Model Card

See [here](./model_card.md).

## CI/CD

This repository includes a CI/CD pipeline using GitHub Actions.
This pipeline runs `flake8`, runs all tests using `pytest`, and if all checks pass the main branch will be deployed to Render by submitting a POST request to a deploy hook.
N.B. this deploy hook is secret and should be defined under the secret `RENDER_DEPLOY_HOOK` in Github.
