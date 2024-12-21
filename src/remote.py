import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dagshub

# # Set the tracking URI to DagsHub
# mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")


import dagshub
dagshub.init(repo_owner='Tejaswini-davu', repo_name='MLFLOW_1', mlflow=True)

# Start an MLflow run
with mlflow.start_run():

    # Load data and split
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and the model
    mlflow.log_param("solver", "liblinear")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # Optionally, log artifacts (e.g., dataset, training logs)
    import pandas as pd
    pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
    mlflow.log_artifact("X_train.csv")
    pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
    mlflow.log_artifact("X_test.csv")
