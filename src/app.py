import logging
import time
import uuid

import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from src.drift import DriftDetector
from src.logging_config import setup_logging
from src.monitoring import Monitor

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "breast-cancer-classifier"
MODEL_ALIAS = "champion"
SLOW_MS = 200

logger = setup_logging("ml_service")
monitor = Monitor()
drift = DriftDetector()
FEATURES = drift.features
drift_state = {"detected": False}


def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    # warm up so the first real request is not slow
    model.predict_proba(drift.reference.head(1))
    logger.info(
        f"Loaded model name={MODEL_NAME} alias={MODEL_ALIAS} version={mv.version}"
    )
    return model, mv.version


model, model_version = load_model()
app = Flask(__name__)


def validate(payload):
    if not isinstance(payload, dict) or "instances" not in payload:
        raise ValueError("Body must be JSON with an 'instances' list")
    rows = payload["instances"]
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("'instances' must be a non empty list")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"instance {i} must be a JSON object")
        missing = [f for f in FEATURES if f not in row]
        if missing:
            raise ValueError(
                f"instance {i} is missing {len(missing)} features, e.g. {missing[:3]}"
            )
        for f in FEATURES:
            v = row[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"instance {i} feature '{f}' must be a number")
    return pd.DataFrame(rows)[FEATURES]


def update_drift_state():
    result = drift.check()
    if result["status"] != "ok":
        return result
    logger.debug(
        f"drift_check share={result['drift_share']} detected={result['drift_detected']}"
    )
    if result["drift_detected"] and not drift_state["detected"]:
        logger.warning(
            f"DATA DRIFT DETECTED drifted={result['drifted_features']}/"
            f"{result['total_features']} share={result['drift_share']} "
            f"top={result['top_drifted']}"
        )
    elif not result["drift_detected"] and drift_state["detected"]:
        logger.info("Data drift cleared, live inputs look like training again")
    drift_state["detected"] = result["drift_detected"]
    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": model_version,
    })


@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()
    rid = uuid.uuid4().hex[:8]
    try:
        df = validate(request.get_json(silent=True))
    except ValueError as e:
        latency = (time.perf_counter() - start) * 1000
        monitor.record(latency, error=True)
        logger.error(
            f"request_id={rid} endpoint=/predict status=400 "
            f"latency_ms={latency:.2f} error=\"{e}\""
        )
        return jsonify({"error": str(e), "request_id": rid}), 400

    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]
    latency = (time.perf_counter() - start) * 1000
    monitor.record(latency, n_predictions=len(df))

    slow = latency > SLOW_MS
    logger.log(
        logging.WARNING if slow else logging.INFO,
        f"request_id={rid} endpoint=/predict status=200 n={len(df)} "
        f"latency_ms={latency:.2f}" + (" slow_request=true" if slow else ""),
    )
    logger.debug(
        f"request_id={rid} predictions={[int(p) for p in preds]} "
        f"proba_malignant={[round(float(p), 4) for p in proba]}"
    )

    drift.add(df)
    update_drift_state()

    return jsonify({
        "request_id": rid,
        "model_version": model_version,
        "predictions": [
            {
                "prediction": int(p),
                "label": "malignant" if int(p) == 1 else "benign",
                "probability_malignant": round(float(pr), 4),
            }
            for p, pr in zip(preds, proba)
        ],
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    result = drift.check()
    return jsonify({
        **monitor.summary(),
        "model_version": model_version,
        "drift_status": result["status"],
        "drift_detected": result["drift_detected"],
        "drift_share": result.get("drift_share", 0.0),
        "drift_samples": result["samples"],
    })


@app.route("/drift", methods=["GET"])
def drift_report():
    return jsonify(drift.check())


@app.route("/drift/reset", methods=["POST"])
def drift_reset():
    drift.reset()
    drift_state["detected"] = False
    logger.info("Drift buffer reset")
    return jsonify({"status": "reset"})


DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ML Service Monitor</title>
<style>
 body { font-family: Arial, sans-serif; background: #f4f6f8; margin: 30px; }
 h1 { margin-bottom: 4px; }
 .sub { color: #555; margin-bottom: 20px; }
 .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
 .card { background: white; border-radius: 8px; padding: 16px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
 .label { color: #666; font-size: 13px; }
 .value { font-size: 28px; font-weight: bold; margin-top: 6px; }
 .ok { color: #1a7f37; } .bad { color: #cf222e; }
</style>
</head>
<body>
<h1>ML Service Monitor</h1>
<div class="sub">breast-cancer-classifier @champion, version <span id="ver">-</span>,
refreshes every 2 seconds</div>
<div class="grid">
 <div class="card"><div class="label">Request count</div><div class="value" id="req">-</div></div>
 <div class="card"><div class="label">Error count</div><div class="value" id="err">-</div></div>
 <div class="card"><div class="label">Error rate</div><div class="value" id="rate">-</div></div>
 <div class="card"><div class="label">Predictions served</div><div class="value" id="pred">-</div></div>
 <div class="card"><div class="label">Avg latency (ms)</div><div class="value" id="avg">-</div></div>
 <div class="card"><div class="label">p95 latency (ms)</div><div class="value" id="p95">-</div></div>
 <div class="card"><div class="label">Max latency (ms)</div><div class="value" id="max">-</div></div>
 <div class="card"><div class="label">Data drift</div><div class="value" id="drift">-</div></div>
</div>
<script>
async function refresh() {
  const r = await fetch("/metrics");
  const m = await r.json();
  document.getElementById("ver").textContent = m.model_version;
  document.getElementById("req").textContent = m.request_count;
  document.getElementById("err").textContent = m.error_count;
  document.getElementById("rate").textContent = (m.error_rate * 100).toFixed(1) + "%";
  document.getElementById("pred").textContent = m.prediction_count;
  document.getElementById("avg").textContent = m.latency_avg_ms;
  document.getElementById("p95").textContent = m.latency_p95_ms;
  document.getElementById("max").textContent = m.latency_max_ms;
  const d = document.getElementById("drift");
  if (m.drift_status !== "ok") {
    d.textContent = "waiting (" + m.drift_samples + "/30)"; d.className = "value";
  } else if (m.drift_detected) {
    d.textContent = "DRIFT " + (m.drift_share * 100).toFixed(0) + "%"; d.className = "value bad";
  } else {
    d.textContent = "OK " + (m.drift_share * 100).toFixed(0) + "%"; d.className = "value ok";
  }
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template_string(DASHBOARD_HTML)


if __name__ == "__main__":
    logger.info("Starting ML service on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001)
