import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,mean_squared_error , r2_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import seaborn as sns
import mlflow

df =pd.read_csv("/home/tejaswini/Documents/VSCODE/MLOPS/MLflow/train.csv")

mlflow.set_experiment('MLOPS-Exp2')
mlflow.autolog()
with mlflow.start_run():
    mlflow.set_tag('dataset', 'iris')

    
    sns.stripplot(x="ram", y="price_range", data=df)
    plt.savefig("price_ram_graph.png")
    plt.close()
    # mlflow.log_artifact("price_ram_graph.png")

    fig = px.pie(df, names='price_range')
    fig.write_image("plot.png")

    # mlflow.log_artifact("plot.png")

    #phones prices have 4 categories, all categories are equaled in count 500 for each category
    threshold = 0.003

    correlation_matrix = df.corr()

    high_corr_features = correlation_matrix.index[abs(correlation_matrix["price_range"]) > threshold].tolist()

    high_corr_features.remove("price_range")

    X_selected = df[high_corr_features]

    Y = df["price_range"]
    scaler = StandardScaler()

    mlflow.log_param("threshold",threshold)

    X_scaled = scaler.fit_transform(X_selected)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    # Define file paths
    X_train_file = "X_train.csv"
    X_test_file = "X_test.csv"

    # Save the data as CSV files
    # np.savetxt(X_train_file, X_train, delimiter=",")
    # np.savetxt(X_test_file, X_test, delimiter=",")

    # mlflow.log_artifact(X_train_file)
    # mlflow.log_artifact(X_test_file)

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    


    # mlflow.sklearn.log_model(logreg, "LogisticRegression")

    # mlflow.log_params({"max_iter":1000, "random_state":42}) 

    accuracy=accuracy_score(Y_pred, Y_pred)

    conf_matrix=confusion_matrix(Y_test, Y_pred)

    class_report=classification_report(Y_test, Y_pred)

    mlflow.log_metric("accuracy", accuracy)
    
    # with open("confusion_matrix.txt", "w") as f:
    #     f.write(str(conf_matrix))
    # mlflow.log_artifact("confusion_matrix.txt")

    # with open("classification_report.txt", "w") as f:
    #     f.write(str(class_report))
    # mlflow.log_artifact("classification_report.txt")






# #     # print(f"Accuracy: ({accuracy:.4f}")

# #     # print("\nConfusion Matrix:")

# #     # print(conf_matrix)

# #     # print("\nClassification Report:")

# #     # print(class_report)

# #     # plt.figure(figsize=(8, 6))

# #     # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,

# #     #             xticklabels=["Predicted Negative", "Predicted Positive"],

# #     #             yticklabels=["Actual Negative", "Actual Positive"])

# #     # plt.xlabel("Predicted Label")

# #     # plt.ylabel("True Label")

# #     # plt.title("Confusion Matrix Heatmap")

# #     # plt.show()
# #     # print(accuracy_score(Y_test, Y_pred))