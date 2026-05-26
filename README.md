# Revenue Protector ML

An end-to-end Machine Learning system for predicting **customer churn** and identifying revenue risk using telecom customer data.

This project includes:
- Data preprocessing (EDA + feature engineering)
- ML model training with MLflow tracking
- FastAPI backend for predictions
- Streamlit dashboard for visualization
- CI pipeline using GitHub Actions

---

## Problem Statement

Telecom companies lose revenue due to customer churn.  
This system predicts whether a customer is likely to churn so that proactive retention strategies can be applied.

---

## Solution Overview

The system follows an end-to-end ML lifecycle:

Raw Data → EDA → Cleaned Data → Model Training → MLflow Tracking → API → Dashboard


---

## Project Structure

```

revenue-protector/
│
├── data/                # Raw and cleaned datasets
├── notebooks/          # EDA notebook
├── model/              # Training + pipeline + evaluation
├── app/                # FastAPI backend (predict + explainability)
├── dashboard/          # Streamlit UI
├── mlruns/             # MLflow tracking
├── requirements.txt
└── README.md

## Tech Stack

Python
Pandas, NumPy
Scikit-learn
XGBoost
MLflow (Experiment tracking)
FastAPI (Backend API)
Streamlit (Dashboard)
SHAP (Explainability)
GitHub Actions (CI/CD)

## How to Run the Project
1 Clone the repository
git clone https://github.com/YOUR_USERNAME/revenue-protector-ml.git
cd revenue-protector-ml
2 Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3 Install dependencies
pip install -r requirements.txt
4 Run EDA (optional)
jupyter notebook notebooks/eda.ipynb
5 Train the model
python model/train.py
6 Run FastAPI backend
uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs
7 Run Streamlit dashboard
streamlit run dashboard/app.py

8 MLflow Tracking

To view experiments:

mlflow ui

Open:

http://127.0.0.1:5000

## Features
Telecom Customer churn prediction
Model tracking with MLflow
REST API for real-time inference
SHAP-based model explainability
Interactive dashboard

CI/CD Pipeline:
GitHub Actions automatically:
    Installs dependencies
    Validates imports
    Ensures code integrity on every push

Future Improvements:
Docker containerization
Cloud deployment (AWS / Azure)
Automated retraining pipeline
Data drift detection
Authentication layer for API
