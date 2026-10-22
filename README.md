# DAM405 Programming Assignment 3: From AutoML to a Monitored, Explainable Model

This project trains and tunes a breast cancer classifier with FLAML AutoML, compares it with a Logistic Regression baseline, tracks every experiment in MLflow, registers the best model in the MLflow Model Registry, explains it with SHAP, and serves the registered model through a Flask API with structured logging, monitoring and a data drift check.

**Student:** Tenzin Namgay (Eyemusican)
**Module:** DAM405 Machine Learning Operations, College of Science and Technology

## Dataset

Breast Cancer Wisconsin (Diagnostic) dataset from scikit learn: 569 patients, 30 numeric features.
Labels are flipped so that **1 = malignant** and **0 = benign**, which makes recall mean "how many cancer cases were caught".
Split: 80% train (455 rows), 20% test (114 rows), stratified, `random_state=42`.

## Project structure

```
dam405-assignment3-automl-monitoring/
├── src/
│   ├── __init__.py
│   ├── data.py              # load and split the dataset
│   ├── evaluate.py          # accuracy, precision, recall, f1, roc_auc
│   ├── train.py             # baseline + FLAML AutoML, MLflow tracking, Model Registry
│   ├── explain.py           # SHAP explainability on the registered model
│   ├── logging_config.py    # console + rotating file logging
│   ├── monitoring.py        # request count, error rate, latency (avg, p95, max)
│   ├── drift.py             # KS test drift detector
│   ├── app.py               # Flask service loading the @champion model
│   └── simulate.py          # sends normal, broken and drifted traffic
├── tests/
│   ├── __init__.py
│   └── test_core.py         # pytest unit tests
├── reports/                 # comparison table, confusion matrices, SHAP plots, FLAML log
├── logs/
│   └── service.log          # sample log output
├── evidence/                # screenshots used in the report
├── mlruns/                  # MLflow artifacts
├── mlflow.db                # MLflow run store and Model Registry
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python 3.14 on Windows was used.

~~~powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
~~~

## How to run

**1. Train and track (baseline vs AutoML, registers the best model)**

~~~powershell
python -m src.train
~~~

**2. Open the MLflow UI** (switch the sidebar to "Model training")

~~~powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
~~~

**3. Explain the registered model with SHAP**

~~~powershell
python -m src.explain
~~~

**4. Start the service** (port 5001)

~~~powershell
python -m src.app
~~~

**5. Send traffic** (in another terminal)

~~~powershell
python -m src.simulate single
python -m src.simulate normal
python -m src.simulate errors
python -m src.simulate drift
python -m src.simulate metrics
~~~

Live dashboard: http://127.0.0.1:5001/dashboard

**6. Run tests**

~~~powershell
python -m pytest -v
~~~

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service status and model version |
| POST | `/predict` | Predictions for `{"instances": [ {30 features}, ... ]}` |
| GET | `/metrics` | Request count, error rate, latency, drift status |
| GET | `/drift` | Full drift report |
| POST | `/drift/reset` | Clears the drift window |
| GET | `/dashboard` | Live monitoring page, refreshes every 2 seconds |

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Train time |
|---|---|---|---|---|---|---|
| Baseline Logistic Regression | 0.9649 | 0.975 | 0.9286 | 0.9512 | 0.996 | 0.02 s |
| FLAML AutoML (LightGBM) | 0.9737 | 1.000 | 0.9286 | 0.9630 | 0.995 | 60 s |

The AutoML model was registered as `breast-cancer-classifier` version 1 with the alias `champion`.

## Logging and monitoring

* **Levels:** DEBUG (predictions and probabilities), INFO (normal requests), WARNING (slow requests over 200 ms, data drift), ERROR (invalid input)
* **Handlers:** console at INFO, rotating file `logs/service.log` at DEBUG (1 MB, 3 backups)
* **Format:** `timestamp | level | logger | key=value message`, with a `request_id` on every request
* **Signals:** request count, error count, error rate, predictions served, average, p95 and max latency
* **Drift:** the last 200 inputs are compared with the training data using a two sample KS test per feature. Drift is flagged when at least 30% of features have p < 0.05, with a minimum of 30 samples.

## Libraries used

FLAML, LightGBM, MLflow, SHAP, scikit learn, pandas, NumPy, SciPy, Flask, matplotlib, pytest, requests.

## References

* MLflow documentation: https://mlflow.org/docs/latest/index.html
* SHAP documentation: https://shap.readthedocs.io
* FLAML documentation: https://microsoft.github.io/FLAML/
* Breast Cancer Wisconsin dataset: https://scikit-learn.org/stable/datasets/toy_dataset.html
