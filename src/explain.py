import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

from src.data import load_data, split_data

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "dam405-assignment3"
MODEL_URI = "models:/breast-cancer-classifier@champion"


def get_explanation(model, X_train, X_test):
    estimator = getattr(getattr(model, "model", None), "estimator", None)
    try:
        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(X_test)
        method = "TreeExplainer"
    except Exception as e:
        print("TreeExplainer failed, using model agnostic Explainer:", e)
        background = shap.sample(X_train, 100, random_state=42)

        def predict_malignant(data):
            df = pd.DataFrame(data, columns=X_test.columns)
            return model.predict_proba(df)[:, 1]

        explainer = shap.Explainer(predict_malignant, background)
        explanation = explainer(X_test)
        method = "Explainer (model agnostic)"
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    return explanation, method


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    print("Loading model from registry:", MODEL_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    explanation, method = get_explanation(model, X_train, X_test)
    print("SHAP method:", method)

    files = []

    shap.plots.beeswarm(explanation, max_display=10, show=False)
    save("reports/shap_summary.png")
    files.append("reports/shap_summary.png")

    shap.plots.bar(explanation, max_display=10, show=False)
    save("reports/shap_bar.png")
    files.append("reports/shap_bar.png")

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv("reports/shap_importance.csv", index=False)
    files.append("reports/shap_importance.csv")
    print("\nTop 10 features by mean |SHAP|:")
    print(importance.head(10).round(4).to_string(index=False))

    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))
    caught = np.where((y_true == 1) & (y_pred == 1))[0]
    missed = np.where((y_true == 1) & (y_pred == 0))[0]
    print(f"\nMalignant cases caught: {len(caught)}, missed: {len(missed)}")

    for name, rows in [("caught", caught), ("missed", missed)]:
        if len(rows) == 0:
            continue
        idx = rows[0]
        shap.plots.waterfall(explanation[idx], max_display=10, show=False)
        path = f"reports/shap_waterfall_{name}.png"
        save(path)
        files.append(path)
        print(f"Waterfall saved for {name} malignant case (test row {idx})")

    with mlflow.start_run(run_name="shap_explainability"):
        mlflow.log_params({"model_uri": MODEL_URI, "shap_method": method})
        for f in files:
            mlflow.log_artifact(f)
    print("\nSHAP plots logged to MLflow run 'shap_explainability'")


if __name__ == "__main__":
    main()
