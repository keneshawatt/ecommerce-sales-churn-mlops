# Demo Script: ingestion to dashboard to model (about 12 minutes)

The assignment says the demo must show the pipeline from data ingestion to dashboard output.
This order covers that, then Part 2.

## Before the demo (15 minutes earlier)

Open four terminal tabs in the project folder (`cd ~/Documents/ecommerce_MLOPS`):

| Tab | Command | Leave it running |
|-----|---------|------------------|
| A | `open -a Docker`, wait for the whale icon to settle, then `docker compose up` | yes |
| B | `airflow standalone` (then open http://localhost:8080) | yes |
| C | `/opt/anaconda3/bin/mlflow ui --backend-store-uri sqlite:////Users/keneshawatt/Documents/ecommerce_MLOPS/mlflow.db --port 5001` | yes |
| D | free for commands | no |

Then, in tab D, **wake the database** (the free Neon tier sleeps when idle, and the first query
is slow):

```bash
/opt/anaconda3/bin/python scripts/check_counts.py
```

Keep these open in the browser: Airflow (8080), the Neon console, the dashboard (8501), the API
docs (8000/docs), MLflow (5001), and `docs/architecture_diagram.png`.

Do **not** trigger the full 13-month DAG live. It takes a long time. Show the last successful run.

## The demo

**1. Problem and architecture (1 min).** Show the architecture diagram. Say: raw retail data goes
through a monthly, Airflow-driven ETL into a PostgreSQL star schema, feeds a dashboard, and then
a churn model that is served, containerised and monitored.

**2. Source and ingestion (1 min).**
```bash
ls data/landing
tail -3 data/logs/ingestion_log.jsonl
```
Point out: data fetched through the UCI API, an untouched raw copy kept, one landing file per
month, and every run logged with its timestamp, status and row count.

**3. Airflow (2 min).** Open the `ecommerce_monthly_etl` DAG. Show the graph with every task
green, open one task's log, and explain that each task processes one monthly file in order.

*Optional idempotency demo (only after you have run `scripts/make_etl_idempotent.py`):* run
`check_counts.py`, clear one task in Airflow so it reruns, run `check_counts.py` again, and show
the numbers did not change.

**4. Warehouse and data quality (2 min).** Run `scripts/check_counts.py` and show:
- the table counts
- the reconciliation between landing rows and `fact_sales` plus rejected rows
- the rejected rows by reason
- the validation proof (no cancelled orders, no non-positive quantities or prices in the fact table)

Then open the Neon console and show the `fact_sales` and dimension tables.

**5. Dashboard (1.5 min).** Open http://localhost:8501. Show the revenue trend, top products,
country sales, RFM segmentation and repeat-purchase views, and use the country filter.

**6. Model tracking and registry (1.5 min).** Open MLflow (5001). Show the three runs side by
side (Logistic Regression, Random Forest, XGBoost), then the registered `churn-model`. Explain the
chronological split and why Logistic Regression was chosen.

**7. Prediction API (1 min).** Open http://localhost:8000/health (it shows the model version and
that it was loaded from the exported folder). In `/docs`, send these two requests to `/predict`:

At-risk customer (expect a high churn probability):
```json
{"recency_days": 300, "frequency": 1, "monetary": 40, "avg_order_value": 40, "tenure_days": 300, "orders_last_90d": 0}
```

Loyal customer (expect a low churn probability):
```json
{"recency_days": 5, "frequency": 12, "monetary": 2500, "avg_order_value": 208, "tenure_days": 330, "orders_last_90d": 4}
```

**8. Churn view in the dashboard (1 min).** Scroll to section 6. Pick a real customer to show
the computed features and the model's prediction, then use the what-if tab.

**9. Monitoring (2 min).** In tab D:
```bash
/opt/anaconda3/bin/python mlops/simulate_traffic.py --n 200
/opt/anaconda3/bin/python mlops/monitor.py --window 200
/opt/anaconda3/bin/python mlops/simulate_traffic.py --n 200 --drift
/opt/anaconda3/bin/python mlops/monitor.py --window 200
```
The first report shows no alerts. The second shows drift alerts and RETRAIN RECOMMENDED. Mention
that the performance table does not change between the two runs, because it re-scores historical
periods with known outcomes.

**10. Docker (30 seconds).** Run `docker ps` and show the `api` and `dashboard` containers.

**11. Lifecycle and limitations (1 min).** Show `docs/model_lifecycle.md`: retrain when two or
more features drift (PSI above 0.25), or precision or recall falls below 0.60, or on a schedule.
Mention the limitations honestly (one year of data, modest AUC of about 0.73, no customer IDs in
the prediction log).

## If something goes wrong

| Problem | Fix |
|---------|-----|
| First database query is very slow | Neon was asleep. Wait 10 seconds and rerun. Warming it up first avoids this |
| `docker` command not found or cannot connect | Docker Desktop is not running. Run `open -a Docker` and wait |
| Dashboard shows a connection error | Check `.env` exists and has `DATABASE_URL`. Restart with `docker compose up` |
| Churn view says it cannot reach the API | `docker compose ps`, then `docker compose logs api` |
| Airflow will not start | Show your saved screenshots of the green DAG run instead |
| Port already in use | Stop any local `uvicorn` or `streamlit` still running (Ctrl+C in its tab) |

After the demo, reset your Neon database password, since it was shared in chat several times
during development.
