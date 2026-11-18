
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- Churn definition and chronological split (dataset ends 2011-12-09) ----
CHURN_WINDOW_DAYS = 60
CUTOFFS = {"train": "2011-06-09", "val": "2011-08-09", "test": "2011-10-09"}
FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value",
            "tenure_days", "orders_last_90d"]
REGISTERED_MODEL_NAME = "churn-model"


def load_transactions() -> pd.DataFrame:
    engine = create_engine(
        os.environ["DATABASE_URL"],
        pool_pre_ping=True,
        connect_args={"connect_timeout": 15},
    )
    # Aggregate to one row per invoice inside the database: far less data to download,
    # and identical features (frequency counts invoices, monetary sums revenue).
    query = """
        SELECT c.customer_id, d.full_date, f.invoice_no, SUM(f.revenue) AS revenue
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_key = c.customer_key
        JOIN dim_date d     ON f.date_key = d.date_key
        GROUP BY c.customer_id, d.full_date, f.invoice_no
    """
    print("Querying Neon...", flush=True)
    tx = pd.read_sql(query, engine)
    tx["full_date"] = pd.to_datetime(tx["full_date"])
    tx["revenue"] = tx["revenue"].astype(float)
    return tx


def build_snapshot(tx: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """One row per customer active on/before cutoff, with a churn label."""
    cutoff = pd.Timestamp(cutoff)
    past = tx[tx["full_date"] <= cutoff]
    future = tx[(tx["full_date"] > cutoff)
                & (tx["full_date"] <= cutoff + pd.Timedelta(days=CHURN_WINDOW_DAYS))]

    g = past.groupby("customer_id")
    snap = pd.DataFrame({
        "last_purchase": g["full_date"].max(),
        "first_purchase": g["full_date"].min(),
        "frequency": g["invoice_no"].nunique(),
        "monetary": g["revenue"].sum(),
    })
    snap["recency_days"] = (cutoff - snap["last_purchase"]).dt.days
    snap["tenure_days"] = (cutoff - snap["first_purchase"]).dt.days
    snap["avg_order_value"] = snap["monetary"] / snap["frequency"]

    recent = past[past["full_date"] > cutoff - pd.Timedelta(days=90)]
    snap["orders_last_90d"] = (recent.groupby("customer_id")["invoice_no"].nunique()
                               .reindex(snap.index).fillna(0))

    snap["churn"] = (~snap.index.isin(future["customer_id"].unique())).astype(int)
    return snap[FEATURES + ["churn"]]


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    tx = load_transactions()
    print(f"Loaded {len(tx):,} transaction rows, "
          f"{tx['full_date'].min().date()} to {tx['full_date'].max().date()}")

    splits = {name: build_snapshot(tx, cutoff) for name, cutoff in CUTOFFS.items()}
    for name, df in splits.items():
        print(f"{name}: {len(df):,} customers, churn rate {df['churn'].mean():.1%}")
    X_train, y_train = splits["train"][FEATURES], splits["train"]["churn"]
    X_val, y_val = splits["val"][FEATURES], splits["val"]["churn"]
    X_test, y_test = splits["test"][FEATURES], splits["test"]["churn"]

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42),
    }

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("churn-prediction")

    best = {"name": None, "run_id": None, "val_auc": -1.0}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            val_metrics = evaluate(model, X_val, y_val)

            mlflow.log_param("model_type", name)
            mlflow.log_param("churn_window_days", CHURN_WINDOW_DAYS)
            mlflow.log_param("cutoffs", str(CUTOFFS))
            mlflow.log_param("features", ",".join(FEATURES))
            mlflow.log_params({f"hp_{k}": v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool)) and k != "steps"}
                              if name != "logistic_regression" else {})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")

            print(f"{name}: val {val_metrics}")
            if val_metrics["roc_auc"] > best["val_auc"]:
                best = {"name": name, "run_id": run.info.run_id,
                        "val_auc": val_metrics["roc_auc"]}

    # Final, unbiased check of the winner on the held-out test period, then register it
    winner = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    test_metrics = evaluate(winner, X_test, y_test)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", REGISTERED_MODEL_NAME)

    print(f"\nBest model: {best['name']} (val ROC-AUC {best['val_auc']:.3f})")
    print(f"Test metrics: {test_metrics}")
    print(f"Registered as '{REGISTERED_MODEL_NAME}' version {version.version}")


if __name__ == "__main__":
    main()