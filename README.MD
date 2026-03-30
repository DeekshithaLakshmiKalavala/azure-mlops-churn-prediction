# Azure MLOps Pipeline for Customer Churn Prediction

## Overview
This project is an end-to-end Azure MLOps pipeline for customer churn prediction. It includes model training, API deployment, Docker containerization, CI automation with GitHub Actions, and Azure ML online endpoint deployment.

## Features
- Customer churn prediction using scikit-learn
- FastAPI REST API for inference
- Dockerized application
- Automated CI pipeline with GitHub Actions
- Azure ML managed online endpoint deployment

## Tech Stack
- Python
- pandas
- scikit-learn
- FastAPI
- Docker
- GitHub Actions
- Azure Machine Learning

## Project Structure
```text
app/              # FastAPI application
src/              # Training scripts
tests/            # API tests
deployment/       # Azure deployment configs
models/           # Saved trained model
data/             # Dataset