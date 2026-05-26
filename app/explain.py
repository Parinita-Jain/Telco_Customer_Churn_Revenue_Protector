import shap
import pandas as pd

def explain(data):

    from app.model import model

    pre = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]
    explainer = shap.TreeExplainer(rf)

    df = pd.DataFrame([data.dict()])
    X_transformed = pre.transform(df)

    shap_values = explainer.shap_values(X_transformed)

    features = pre.get_feature_names_out()

    values = shap_values[0]

    explanation = {}

    for i, val in enumerate(values):
        explanation[features[i]] = float(val)

    return {"shap_values": explanation}