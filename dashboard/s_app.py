import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.predict import predict

st.title("Revenue Protector Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    results = []

    for _, row in df.iterrows():
        result = predict(row)
        results.append(result)

    df["Churn_Risk"] = [r["risk"] for r in results]
    df["Probability"] = [r["probability"] for r in results]

    st.dataframe(df)

    st.bar_chart(df["Churn_Risk"].value_counts())