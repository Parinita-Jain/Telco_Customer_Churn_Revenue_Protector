from fastapi import FastAPI
from app.schema import Customer
from app.predict import predict
from app.explain import explain

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Revenue Protector API running"}

@app.post("/predict")
def get_prediction(data: Customer):
    return predict(data)

@app.post("/explain")
def get_explanation(data: Customer):
    return explain(data)