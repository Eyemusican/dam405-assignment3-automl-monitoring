import math
import threading
import time
from collections import deque


class Monitor:
    """Keeps simple in-memory monitoring signals for the service."""

    def __init__(self, max_latencies=1000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.prediction_count = 0
        self.latencies_ms = deque(maxlen=max_latencies)

    def record(self, latency_ms, error=False, n_predictions=0):
        with self._lock:
            self.request_count += 1
            self.latencies_ms.append(latency_ms)
            if error:
                self.error_count += 1
            self.prediction_count += n_predictions

    def summary(self):
        with self._lock:
            lat = sorted(self.latencies_ms)
            n = len(lat)
            return {
                "uptime_sec": round(time.time() - self.start_time, 1),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": round(self.error_count / self.request_count, 4)
                if self.request_count else 0.0,
                "prediction_count": self.prediction_count,
                "latency_avg_ms": round(sum(lat) / n, 2) if n else 0.0,
                "latency_p95_ms": round(lat[math.ceil(0.95 * n) - 1], 2) if n else 0.0,
                "latency_max_ms": round(lat[-1], 2) if n else 0.0,
            }
