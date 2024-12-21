import optuna
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import mlflow.sklearn

# Load dataset (you can replace this with your own dataset)
data = load_iris()
X, y = data.data, data.target

# Define objective function for Optuna optimization
def objective(trial):
    # Suggest values for hyperparameters
    C = trial.suggest_loguniform('C', 1e-5, 1e2)
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    # Create the Logistic Regression model
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    
    # Start a new MLflow run for this trial
    with mlflow.start_run():
        # Start a nested run for logging the parameter 'C' with its value for this trial
        with mlflow.start_run(nested=True):
            mlflow.log_param('C', C)  # Log the parameter 'C'
            mlflow.log_param('max_iter', max_iter)  # Log the parameter 'max_iter'
        
        # Perform cross-validation (or any other evaluation metric)
        score = cross_val_score(model, X, y, cv=3).mean()

        # Log the performance metric
        mlflow.log_metric('accuracy', score)

        # Log the model
        mlflow.sklearn.log_model(model, "model")
    
    return score

# Set the experiment name
mlflow.set_experiment('optuna_hyperparameter_tuning')

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Print the best trial result
best_trial = study.best_trial
print(f"Best trial:")
print(f"  Value: {best_trial.value}")
print(f"  Params: {best_trial.params}")
