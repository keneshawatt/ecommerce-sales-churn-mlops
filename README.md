# E-Commerce Sales Analytics and Customer Churn Prediction

Data Engineering and MLOps individual project. **Part 1** is a retail data pipeline and
dashboard: UCI Online Retail data goes through Airflow-orchestrated ETL into a PostgreSQL
(Neon) star schema and is shown in a Streamlit dashboard. **Part 2** adds a customer churn
model with MLflow tracking and registry, a FastAPI prediction service, Docker packaging,
monitoring, and a documented retraining plan.

## Architecture

```
UCI Online Retail (ucimlrepo API)
  -> ingest.py: raw copy + monthly landing files (data/raw, data/landing)
  -> Airflow DAG (ecommerce_monthly_etl): one ETL task per monthly file
  -> etl.py: clean, validate, stage, load (rejected rows go to etl_error_log)
  -> Neon PostgreSQL star schema: dim_customer, dim_product, dim_geography, dim_date,
     fact_sales, fact_customer_retention
  -> sales_mart and customer_mart views
  -> Streamlit dashboard (docker)

Part 2 (MLOps layer)
  -> mlops/train.py: churn label, chronological split, 3 models, MLflow tracking + registry
  -> mlops/export_model.py: registered model exported to models/churn-model
  -> mlops/api.py: FastAPI service (docker), logs every prediction
  -> dashboard churn-risk view calls the API
  -> mlops/monitor.py: drift, data quality, performance, latency, retraining decision
```

The architecture diagram is in `docs/architecture_diagram.png`, the data dictionary and
validation rules in `docs/data_dictionary.md`, and the model lifecycle and retraining plan in
`docs/model_lifecycle.md`.

## Project structure

```
ingest.py                    ingestion: raw copy + monthly landing files
etl.py                       clean, validate, load to the star schema
dags/ecommerce_etl_dag.py    Airflow DAG (one task per monthly file)
scripts/fetch_data.py        fetch the dataset through the UCI API
scripts/build_marts_and_retention.py   create the marts, fill fact_customer_retention
sql/schema.sql               star schema DDL
sql/marts.sql                sales_mart and customer_mart views
dashboard/app.py             Streamlit dashboard (6 views, including churn risk)
mlops/train.py               feature engineering, training, MLflow tracking and registry
mlops/export_model.py        export the registered model for Docker
mlops/api.py                 FastAPI prediction service
mlops/simulate_traffic.py    send test customers to the API
mlops/monitor.py             monitoring and retraining decision
Dockerfile.api, Dockerfile.dashboard, docker-compose.yml, .dockerignore
docs/                        data dictionary, model lifecycle, architecture diagram
```

## Setup

1. **Python packages** (Python 3.13 was used):
   ```bash
   pip install pandas sqlalchemy psycopg2-binary python-dotenv ucimlrepo streamlit plotly \
       requests apache-airflow scikit-learn xgboost mlflow fastapi uvicorn scipy
   ```
2. **Database.** Create a free project on [Neon](https://neon.tech) and copy its connection string.
3. **Credentials.** Create a `.env` file in the project root. It is listed in `.gitignore` and
   `.dockerignore` and must never be committed:
   ```
   DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST/DBNAME?sslmode=require
   ```
   Credentials are read from environment variables everywhere. None are hardcoded.
4. **Tables.** Run `sql/schema.sql` in the Neon SQL editor (`etl.py` also creates any missing
   tables).

## Part 1: run the pipeline

```bash
# 1. Fetch the dataset (raw copy saved to data/raw/)
python scripts/fetch_data.py

# 2. Ingest: raw copy + monthly landing files in data/landing/
python ingest.py --source data/raw/online_retail_latest.csv
```

3. **ETL with Airflow.** Edit `PROJECT_ROOT` and `PYTHON_BIN` at the top of
   `dags/ecommerce_etl_dag.py` to match your machine, then:
   ```bash
   cp dags/ecommerce_etl_dag.py ~/airflow/dags/
   airflow standalone
   ```
   Open http://localhost:8080 and trigger the `ecommerce_monthly_etl` DAG. Each monthly file
   becomes one task, chained in order. Failed rows are written to `etl_error_log` and each run
   to `etl_run_log`.

4. **Marts and retention fact table:**
   ```bash
   python scripts/build_marts_and_retention.py
   ```
5. **Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

**Verify the load.** `fact_sales` plus `etl_error_log` should account for the rows in the
landing files (541,909 for the full dataset). A clean run gives 387,841 rows in `fact_sales`.

## Part 2: model, API, containers, monitoring

```bash
# Train three models, track them in MLflow, register the best as "churn-model"
python mlops/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # http://localhost:5001

# Export the registered model, then build and start the API and dashboard containers
python mlops/export_model.py
docker compose up --build
```

- API docs and test page: http://localhost:8000/docs (health check at `/health`)
- Dashboard (with the churn-risk view): http://localhost:8501

**Monitoring:**
```bash
python mlops/simulate_traffic.py --n 200            # normal traffic
python mlops/monitor.py --window 200                # expect no alerts
python mlops/simulate_traffic.py --n 200 --drift    # shifted traffic
python mlops/monitor.py --window 200                # expect drift alerts and RETRAIN RECOMMENDED
```

## Results

| Model | Val precision | Val recall | Val F1 | Val ROC-AUC |
|-------|---------------|------------|--------|-------------|
| Logistic Regression (registered) | 0.732 | 0.752 | 0.742 | 0.739 |
| Random Forest | 0.708 | 0.819 | 0.759 | 0.739 |
| XGBoost | 0.683 | 0.874 | 0.767 | 0.702 |

Held-out test (registered model): precision 0.670, recall 0.730, F1 0.699, ROC-AUC 0.734.

## Design decisions

- **Neon (cloud PostgreSQL)** instead of a local database: always reachable from Airflow and
  from containers, with no local setup. The free tier suspends when idle, so the engines use
  `pool_pre_ping` and `pool_recycle`.
- **Monthly landing files** simulate incremental ingestion; unique constraints plus
  `ON CONFLICT DO NOTHING` make reruns safe.
- **Rejected rows are logged, not dropped**, so every data quality decision can be audited.
- **Chronological split** with features computed only from data before each scoring date, to
  avoid leakage.
- **Exported model folder** so the container does not need access to the local MLflow database.

## Known limitations

- One year of data from a single retailer, with modest model performance (ROC-AUC about 0.73).
- The prediction log has no customer IDs, so live precision and recall cannot be computed from
  it. Performance is checked by re-scoring recent periods with known outcomes.
- Monitoring is run manually. Scheduling it weekly with Airflow is a proposed extension.
- Dataset source: UCI Online Retail, Chen, D. (2015), https://doi.org/10.24432/C5BW33
  (CC BY 4.0).
