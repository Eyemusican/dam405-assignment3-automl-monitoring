import json
import sys

import requests

from src.data import load_data, split_data

BASE = "http://127.0.0.1:5001"


def rows(df):
    return df.to_dict("records")


def single(X_test, y_test):
    row = X_test.iloc[[1]]
    r = requests.post(f"{BASE}/predict", json={"instances": rows(row)})
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    print("True label:", "malignant" if int(y_test.iloc[1]) == 1 else "benign")


def normal(X_test, n=40):
    ok = 0
    for i in range(n):
        r = requests.post(f"{BASE}/predict", json={"instances": rows(X_test.iloc[[i]])})
        ok += r.status_code == 200
    print(f"Sent {n} normal requests, {ok} returned 200")


def errors(X_test):
    good_row = rows(X_test.iloc[[0]])[0]
    bad_requests = [
        ("missing features", {"instances": [{"mean radius": 14.0}]}),
        ("wrong type", {"instances": [{**good_row, "mean radius": "big"}]}),
        ("empty list", {"instances": []}),
        ("no instances key", {"data": [1, 2, 3]}),
    ]
    for name, body in bad_requests:
        r = requests.post(f"{BASE}/predict", json=body)
        print(f"{name:18} -> {r.status_code} {r.json()['error']}")
    r = requests.post(f"{BASE}/predict", data="hello", headers={"Content-Type": "text/plain"})
    print(f"{'not JSON':18} -> {r.status_code} {r.json()['error']}")


def drift(X_test, n=40):
    requests.post(f"{BASE}/drift/reset")
    shifted = X_test * 1.5
    for i in range(n):
        requests.post(f"{BASE}/predict", json={"instances": rows(shifted.iloc[[i]])})
    print(f"Sent {n} shifted requests (all values x1.5)")
    print(json.dumps(requests.get(f"{BASE}/drift").json(), indent=2))


def metrics():
    print(json.dumps(requests.get(f"{BASE}/metrics").json(), indent=2))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    if mode == "single":
        single(X_test, y_test)
    elif mode == "normal":
        normal(X_test)
    elif mode == "errors":
        errors(X_test)
    elif mode == "drift":
        drift(X_test)
    elif mode == "metrics":
        metrics()
    else:
        print("Modes: single, normal, errors, drift, metrics")


if __name__ == "__main__":
    main()
