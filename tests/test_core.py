import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, split_data
from src.drift import DriftDetector
from src.evaluate import evaluate
from src.monitoring import Monitor


def test_load_data_shape_and_labels():
    X, y = load_data()
    assert X.shape == (569, 30)
    # 1 = malignant after flipping
    assert int(y.sum()) == 212
    assert int((y == 0).sum()) == 357


def test_split_is_stratified():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 455 and len(X_test) == 114
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_evaluate_returns_valid_metrics():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_monitor_counts_and_error_rate():
    m = Monitor()
    m.record(10, n_predictions=1)
    m.record(20, n_predictions=1)
    m.record(30, error=True)
    m.record(40, n_predictions=1)
    s = m.summary()
    assert s["request_count"] == 4
    assert s["error_count"] == 1
    assert s["error_rate"] == 0.25
    assert s["prediction_count"] == 3
    assert s["latency_avg_ms"] == 25.0
    assert s["latency_p95_ms"] == 40


def _detector(tmp_path):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    ref = tmp_path / "reference.csv"
    X_train.to_csv(ref, index=False)
    return DriftDetector(reference_path=str(ref)), X_test


def test_no_drift_on_normal_data(tmp_path):
    d, X_test = _detector(tmp_path)
    assert d.check(X_test)["drift_detected"] is False


def test_drift_on_shifted_data(tmp_path):
    d, X_test = _detector(tmp_path)
    result = d.check(X_test * 1.5)
    assert result["drift_detected"] is True
    assert result["drifted_features"] >= 10


def test_drift_waits_for_enough_samples(tmp_path):
    d, X_test = _detector(tmp_path)
    d.add(X_test.head(10))
    result = d.check()
    assert result["status"] == "not_enough_data"
    assert result["drift_detected"] is False
