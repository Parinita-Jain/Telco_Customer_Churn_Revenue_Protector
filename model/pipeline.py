import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

def create_pipeline():

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    cat_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]

    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline