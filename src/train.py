import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from flaml import AutoML
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE, load_data, split_data
from src.evaluate import evaluate

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "dam405-assignment3"
MODEL_NAME = "breast-cancer-classifier"
TIME_BUDGET = 60


def save_confusion_matrix(model, X_test, y_test, path, title):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, model.predict(X_test), display_labels=["benign", "malignant"]
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # keep training data as the reference for the drift check later
    X_train.to_csv("reports/train_reference.csv", index=False)

    results = {}

    # ---------- Baseline ----------
    with mlflow.start_run(run_name="baseline_logreg"):
        start = time.time()
        baseline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        )
        baseline.fit(X_train, y_train)
        train_time = time.time() - start

        metrics = evaluate(baseline, X_test, y_test)
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "scaler": "StandardScaler",
            "max_iter": 1000,
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics({**metrics, "train_time_sec": train_time})

        cm_path = "reports/cm_baseline.png"
        save_confusion_matrix(baseline, X_test, y_test, cm_path, "Baseline LogReg")
        mlflow.log_artifact(cm_path)

        info = mlflow.sklearn.log_model(
            baseline,
            name="model",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        results["baseline_logreg"] = (metrics, info.model_uri)
        print("Baseline metrics:", {k: round(v, 4) for k, v in metrics.items()})

    # ---------- AutoML ----------
    with mlflow.start_run(run_name="flaml_automl"):
        automl = AutoML()
        start = time.time()
        automl.fit(
            X_train,
            y_train,
            task="classification",
            metric="roc_auc",
            time_budget=TIME_BUDGET,
            seed=RANDOM_STATE,
            log_file_name="reports/flaml.log",
            mlflow_logging=False,
            verbose=1,
        )
        train_time = time.time() - start

        metrics = evaluate(automl, X_test, y_test)
        mlflow.log_params({
            "model_type": "FLAML AutoML",
            "time_budget": TIME_BUDGET,
            "optimise_metric": "roc_auc",
            "best_estimator": automl.best_estimator,
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_params({f"best_{k}": v for k, v in automl.best_config.items()})
        mlflow.log_metrics({
            **metrics,
            "best_cv_roc_auc": 1 - automl.best_loss,
            "train_time_sec": train_time,
        })

        cm_path = "reports/cm_automl.png"
        save_confusion_matrix(automl, X_test, y_test, cm_path, "FLAML AutoML")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact("reports/flaml.log")

        info = mlflow.sklearn.log_model(
            automl,
            name="model",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        results["flaml_automl"] = (metrics, info.model_uri)
        print("Best estimator:", automl.best_estimator)
        print("AutoML metrics:", {k: round(v, 4) for k, v in metrics.items()})

    # ---------- Compare ----------
    table = pd.DataFrame({k: v[0] for k, v in results.items()}).T.round(4)
    table.to_csv("reports/comparison.csv")
    print("\nComparison on test set:")
    print(table)

    # ---------- Register the best ----------
    best = max(results, key=lambda k: (results[k][0]["f1"], results[k][0]["roc_auc"]))
    print(f"\nBest model by F1 (tie broken by ROC AUC): {best}")

    mv = mlflow.register_model(results[best][1], MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    client.update_model_version(
        MODEL_NAME, mv.version, description=f"Best model from Assignment 3: {best}"
    )
    print(f"Registered {MODEL_NAME} version {mv.version} with alias 'champion'")


if __name__ == "__main__":
    main()
