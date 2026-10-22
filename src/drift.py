import threading
from collections import deque

import pandas as pd
from scipy.stats import ks_2samp


class DriftDetector:
    """Compares recent live inputs with the training data using a KS test per feature."""

    def __init__(
        self,
        reference_path="reports/train_reference.csv",
        p_threshold=0.05,
        share_threshold=0.3,
        min_samples=30,
        window=200,
    ):
        self.reference = pd.read_csv(reference_path)
        self.features = list(self.reference.columns)
        self.p_threshold = p_threshold
        self.share_threshold = share_threshold
        self.min_samples = min_samples
        self._buffer = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, df):
        with self._lock:
            for row in df[self.features].to_dict("records"):
                self._buffer.append(row)

    def reset(self):
        with self._lock:
            self._buffer.clear()

    def check(self, live=None):
        if live is None:
            with self._lock:
                live = pd.DataFrame(list(self._buffer))
        n = len(live)
        if n < self.min_samples:
            return {
                "status": "not_enough_data",
                "samples": n,
                "min_samples": self.min_samples,
                "drift_detected": False,
            }

        results = {}
        for f in self.features:
            stat, p = ks_2samp(self.reference[f], live[f])
            results[f] = {"ks_stat": float(stat), "p_value": float(p)}

        drifted = [f for f, r in results.items() if r["p_value"] < self.p_threshold]
        share = len(drifted) / len(self.features)
        top = sorted(drifted, key=lambda f: results[f]["ks_stat"], reverse=True)[:5]
        return {
            "status": "ok",
            "samples": n,
            "drift_detected": share >= self.share_threshold,
            "drifted_features": len(drifted),
            "total_features": len(self.features),
            "drift_share": round(share, 3),
            "top_drifted": top,
        }
