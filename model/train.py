import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pipeline import create_pipeline

#mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Revenue-Protector")


df = pd.read_csv("data/telco_churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_acc = 0
best_run_id = None

for n in [50, 100, 150]:

    with mlflow.start_run():

        pipeline = create_pipeline()
        pipeline.named_steps["model"].set_params(n_estimators=n)

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)

        # 🔥 only log (not register)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"n={n}, acc={acc}")

        if acc > best_acc:
            best_acc = acc
            best_run_id = mlflow.active_run().info.run_id

print("Best Run:", best_run_id)

# ✅ register ONLY best model
'''
mlflow.register_model(
    f"runs:/{best_run_id}/model",
    "RevenueProtectorModel"
)
'''
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "RevenueProtectorModel"

# Create model if not exists
try:
    client.create_registered_model(model_name)
except:
    pass

# Register best run model
model_uri = f"runs:/{best_run_id}/model"

client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=best_run_id
)

print("✅ Model registered successfully")

mlflow.sklearn.log_model(
    pipeline,
    "model",
    registered_model_name="RevenueProtectorModel"   # 🔥 THIS LINE IS KEY
)



