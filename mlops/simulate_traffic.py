"""
simulate_traffic.py
-------------------
Sends customers to the running prediction API so there is something to monitor.

  normal traffic : real customers resembling the training data (no drift expected)
  --drift        : the same customers with their behaviour shifted (customers who
                   bought long ago, spend less and have no recent orders) to show
                   the drift check firing
  --period test  : real customers from the latest period instead. Expect some drift
                   (especially tenure_days) because customers are simply older by then,
                   which is a genuine reason to retrain periodically.

Run from the project root, with the API container up:
    /opt/anaconda3/bin/python mlops/simulate_traffic.py --n 200
    /opt/anaconda3/bin/python mlops/simulate_traffic.py --n 200 --drift
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import CUTOFFS, FEATURES, build_snapshot, load_transactions  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of requests to send")
    parser.add_argument("--drift", action="store_true", help="send drifted customers")
    parser.add_argument("--period", choices=list(CUTOFFS), default="train",
                        help="which period's customers to send (default: train)")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    args = parser.parse_args()

    snapshot = build_snapshot(load_transactions(), CUTOFFS[args.period])[FEATURES]
    sample = snapshot.sample(n=args.n, replace=args.n > len(snapshot), random_state=42).copy()

    if args.drift:
        sample["recency_days"] = sample["recency_days"] * 2.5
        sample["monetary"] = sample["monetary"] * 0.5
        sample["avg_order_value"] = sample["avg_order_value"] * 0.7
        quiet = sample.sample(frac=0.7, random_state=1).index
        sample.loc[quiet, "orders_last_90d"] = 0

    ok = errors = 0
    with requests.Session() as session:
        for _, row in sample.iterrows():
            payload = {k: float(v) for k, v in row.items()}
            try:
                response = session.post(args.url, json=payload, timeout=10)
                ok += response.status_code == 200
                errors += response.status_code != 200
            except requests.RequestException:
                errors += 1

    kind = "drifted" if args.drift else "normal"
    print(f"Sent {args.n} {kind} requests ({args.period}-period customers): "
          f"{ok} succeeded, {errors} failed.")


if __name__ == "__main__":
    main()
