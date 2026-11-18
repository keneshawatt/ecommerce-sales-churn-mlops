"""
monitor.py
----------
Model monitoring for the churn API. Runs as a batch job and checks three things:

1. Service health   - failures and latency, from logs/predictions.jsonl
2. Input data quality - missing values and values outside the range seen in training
3. Input drift      - are the features the API is receiving still distributed like
                      the training data? (Population Stability Index + KS test)
4. Model performance - precision/recall of the deployed model on recent periods
                      whose churn outcome is now known (labeled windows)

It then applies the retraining criteria below and prints a decision. A JSON copy
of the report is saved to monitoring/report.json.

Run from the project root (the API containers should be up and have served requests):
    /opt/anaconda3/bin/python mlops/monitor.py
    /opt/anaconda3/bin/python mlops/monitor.py --window 200   # only the last 200 requests
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CUTOFFS, FEATURES, build_snapshot, load_transactions  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"
MONITOR_DIR = PROJECT_ROOT / "monitoring"
REFERENCE_PATH = MONITOR_DIR / "reference_features.csv"
REPORT_PATH = MONITOR_DIR / "report.json"
MODEL_DIR = PROJECT_ROOT / "models" / "churn-model"

# ---- Retraining criteria (documented in the model lifecycle) ----
MIN_SAMPLES = 30            # minimum logged predictions before drift is meaningful
PSI_WARN = 0.10             # PSI 0.10-0.25: moderate shift
PSI_ALERT = 0.25            # PSI > 0.25: significant shift
DRIFTED_FEATURES_TO_RETRAIN = 2   # retrain if this many features show significant drift
THRESHOLDS = {"precision": 0.60, "recall": 0.60}   # minimum acceptable on labeled windows
MAX_OUT_OF_RANGE = 0.05     # alert if more than 5% of requests fall outside the training range
MAX_FAILURE_RATE = 0.02     # more than 2% failed requests
MAX_P95_LATENCY_MS = 500    # p95 latency budget
# Labeled evaluation windows: all after the training cutoff, with a full 60-day outcome window
EVAL_CUTOFFS = ["2011-08-09", "2011-09-09", "2011-10-09"]


def load_prediction_log(window: int | None) -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    log = pd.DataFrame(rows)
    if window and len(log) > window:
        log = log.tail(window).reset_index(drop=True)
    return log


def service_health(log: pd.DataFrame) -> dict | None:
    if log.empty:
        return None
    total = len(log)
    failures = int((log["status"] == "failure").sum())
    latency = log["latency_ms"].astype(float)
    health = {
        "requests": int(total),
        "failures": failures,
        "failure_rate": failures / total,
        "latency_mean_ms": float(latency.mean()),
        "latency_p95_ms": float(latency.quantile(0.95)),
        "latency_max_ms": float(latency.max()),
    }
    ok = log[log["status"] == "success"]
    if len(ok):
        health["predicted_churn_rate"] = float(ok["prediction"].astype(float).mean())
        health["mean_churn_probability"] = float(ok["churn_probability"].astype(float).mean())
    return health


def current_features(log: pd.DataFrame) -> pd.DataFrame:
    ok = log[log["status"] == "success"]
    if ok.empty:
        return pd.DataFrame(columns=FEATURES)
    return pd.DataFrame(list(ok["inputs"]))[FEATURES].astype(float)


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference`."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    inner = edges[1:-1]   # outer edges are open-ended so every value falls in a bin

    def proportions(values):
        idx = np.searchsorted(inner, values, side="right")
        counts = np.bincount(idx, minlength=len(inner) + 1).astype(float)
        return np.clip(counts / counts.sum(), 1e-4, None)

    ref_p, cur_p = proportions(reference), proportions(current)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = {}
    for feat in FEATURES:
        value = psi(reference[feat], current[feat])
        ks = ks_2samp(reference[feat], current[feat])
        status = "ALERT" if value > PSI_ALERT else "WARN" if value > PSI_WARN else "ok"
        report[feat] = {"psi": round(value, 4), "ks_statistic": round(float(ks.statistic), 4),
                        "ks_pvalue": float(ks.pvalue), "status": status}
    return report


def input_quality(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Basic data-quality checks on what the API is receiving."""
    quality = {}
    for feat in FEATURES:
        low, high = float(reference[feat].min()), float(reference[feat].max())
        values = current[feat]
        quality[feat] = {
            "missing": int(values.isna().sum()),
            "out_of_range_share": float(((values < low) | (values > high)).mean()),
            "training_min": low,
            "training_max": high,
        }
    return quality


def load_model():
    if MODEL_DIR.exists():
        return mlflow.sklearn.load_model(str(MODEL_DIR))
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    versions = mlflow.MlflowClient().search_model_versions("name='churn-model'")
    latest = max(int(v.version) for v in versions)
    return mlflow.sklearn.load_model(f"models:/churn-model/{latest}")


def performance_report(model, tx: pd.DataFrame) -> list:
    windows = []
    for cutoff in EVAL_CUTOFFS:
        snap = build_snapshot(tx, cutoff)
        X, y = snap[FEATURES], snap["churn"]
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        window = {
            "cutoff": cutoff,
            "customers": int(len(snap)),
            "actual_churn_rate": float(y.mean()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
        }
        below = [k for k, floor in THRESHOLDS.items() if window[k] < floor]
        window["status"] = "ALERT" if below else "ok"
        windows.append(window)
    return windows


def decide(health, quality, drift, performance) -> tuple[list, bool]:
    alerts, retrain = [], False
    if health:
        if health["failure_rate"] > MAX_FAILURE_RATE:
            alerts.append(f"Failure rate {health['failure_rate']:.1%} is above {MAX_FAILURE_RATE:.0%}")
        if health["latency_p95_ms"] > MAX_P95_LATENCY_MS:
            alerts.append(f"p95 latency {health['latency_p95_ms']:.0f} ms is above {MAX_P95_LATENCY_MS} ms")
    if quality:
        for feat, q in quality.items():
            if q["missing"] > 0:
                alerts.append(f"{q['missing']} missing value(s) in {feat}")
            if q["out_of_range_share"] > MAX_OUT_OF_RANGE:
                alerts.append(f"{q['out_of_range_share']:.0%} of requests have {feat} outside the training range")
    if drift:
        drifted = [f for f, v in drift.items() if v["status"] == "ALERT"]
        if drifted:
            alerts.append("Significant input drift in: " + ", ".join(drifted))
        if len(drifted) >= DRIFTED_FEATURES_TO_RETRAIN:
            retrain = True
    for w in performance:
        if w["status"] == "ALERT":
            alerts.append(f"Performance below threshold for window {w['cutoff']} "
                          f"(precision {w['precision']:.2f}, recall {w['recall']:.2f})")
            retrain = True
    return alerts, retrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=None,
                        help="only analyse the last N logged requests")
    args = parser.parse_args()

    MONITOR_DIR.mkdir(exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).isoformat()}

    # 1. Service health
    log = load_prediction_log(args.window)
    health = service_health(log)
    report["service_health"] = health
    print("\n=== 1. Service health (logs/predictions.jsonl) ===")
    if health is None:
        print("No predictions logged yet. Send some requests first (see simulate_traffic.py).")
    else:
        print(f"Requests analysed : {health['requests']}")
        print(f"Failures          : {health['failures']} ({health['failure_rate']:.1%})")
        print(f"Latency (ms)      : mean {health['latency_mean_ms']:.1f} | "
              f"p95 {health['latency_p95_ms']:.1f} | max {health['latency_max_ms']:.1f}")
        if "predicted_churn_rate" in health:
            print(f"Predicted churn   : {health['predicted_churn_rate']:.1%} of requests "
                  f"(mean probability {health['mean_churn_probability']:.2f})")

    tx = load_transactions()

    # Reference = training-period feature distribution (saved on first run)
    if REFERENCE_PATH.exists():
        reference = pd.read_csv(REFERENCE_PATH)
    else:
        reference = build_snapshot(tx, CUTOFFS["train"])[FEATURES]
        reference.to_csv(REFERENCE_PATH, index=False)
        print(f"(saved training reference to {REFERENCE_PATH.relative_to(PROJECT_ROOT)})")
    current = current_features(log)

    # 2. Input data quality
    print("\n=== 2. Input data quality (vs the range seen in training) ===")
    quality = input_quality(reference, current) if len(current) else None
    report["input_quality"] = quality
    if quality is None:
        print("No successful predictions logged yet.")
    else:
        print(f"{'feature':<18}{'missing':>9}{'out of range':>14}   training range")
        for feat, q in quality.items():
            print(f"{feat:<18}{q['missing']:>9}{q['out_of_range_share']:>14.1%}   "
                  f"{q['training_min']:.1f} to {q['training_max']:.1f}")

    # 3. Input drift against the training data
    print("\n=== 3. Input drift vs training data (PSI: <0.10 ok, 0.10-0.25 warn, >0.25 alert) ===")
    drift = None
    if len(current) < MIN_SAMPLES:
        print(f"Only {len(current)} successful predictions logged; need at least {MIN_SAMPLES} "
              f"for a meaningful drift check.")
    else:
        drift = drift_report(reference, current)
        print(f"{'feature':<18}{'PSI':>8}{'KS stat':>10}{'KS p-value':>12}   status")
        for feat, v in drift.items():
            print(f"{feat:<18}{v['psi']:>8.3f}{v['ks_statistic']:>10.3f}{v['ks_pvalue']:>12.4f}   {v['status']}")
    report["drift"] = drift

    # 3. Performance on recent labeled periods
    print("\n=== 4. Model performance on labeled windows (min precision/recall "
          f"{THRESHOLDS['precision']:.2f}/{THRESHOLDS['recall']:.2f}) ===")
    performance = performance_report(load_model(), tx)
    print(f"{'cutoff':<12}{'customers':>10}{'churn':>8}{'precision':>11}{'recall':>8}{'f1':>7}{'auc':>7}   status")
    for w in performance:
        print(f"{w['cutoff']:<12}{w['customers']:>10}{w['actual_churn_rate']:>8.1%}"
              f"{w['precision']:>11.3f}{w['recall']:>8.3f}{w['f1']:>7.3f}{w['roc_auc']:>7.3f}   {w['status']}")
    report["performance"] = performance

    # Decision
    alerts, retrain = decide(health, quality, drift, performance)
    report["alerts"] = alerts
    report["retrain_recommended"] = retrain
    print("\n=== Alerts and retraining decision ===")
    for a in alerts:
        print(f"- {a}")
    if not alerts:
        print("- No alerts.")
    print("RETRAIN RECOMMENDED" if retrain else "No retraining needed.")

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
