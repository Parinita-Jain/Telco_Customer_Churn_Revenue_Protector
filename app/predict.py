import pandas as pd

from app.model import model

def predict(data):

    # ✅ Handle both input types
    if hasattr(data, "dict"):         # FastAPI (Pydantic)
        data = data.dict()
    elif hasattr(data, "to_dict"):   # Pandas row (Streamlit)
        data = data.to_dict()

    df = pd.DataFrame([data])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    risk = "Low"
    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "risk": risk
    }