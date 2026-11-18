"""
retail_pipeline_dag.py
-----------------------
Airflow DAG for the e-commerce ETL pipeline. Demonstrates a repeatable,
scheduled workflow: ingest raw source -> discover landing files ->
run ETL per file -> refresh marts.

Place this file in your Airflow DAGs folder (usually ~/airflow/dags/).
"""
from datetime import datetime, timedelta
import glob
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import ingest_full_file  # noqa: E402
from etl import run_etl  # noqa: E402

default_args = {
    "owner": "student",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ecommerce_sales_pipeline",
    default_args=default_args,
    description="Ingest -> clean -> load -> mart for e-commerce sales data",
    schedule_interval="@daily",     # simulate monthly extracts by running daily in dev
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ecommerce", "assignment-part1"],
) as dag:

    def task_ingest(**kwargs):
        source_path = os.environ.get("RAW_SOURCE_PATH", "data/raw/online_retail.csv")
        ingest_full_file(source_path)

    def task_run_etl_all_landing(**kwargs):
        landing_files = sorted(glob.glob("data/landing/orders_*.csv"))
        if not landing_files:
            raise FileNotFoundError("No landing files found — did ingestion run?")
        for f in landing_files:
            run_etl(f)

    def task_refresh_marts(**kwargs):
        # Placeholder hook: call sql/marts.sql via psycopg2/sqlalchemy here,
        # or trigger a materialized view REFRESH.
        print("Refreshing sales_mart / customer_mart views...")

    ingest_task = PythonOperator(
        task_id="ingest_raw_source",
        python_callable=task_ingest,
    )

    etl_task = PythonOperator(
        task_id="run_etl_on_landing_files",
        python_callable=task_run_etl_all_landing,
    )

    mart_task = PythonOperator(
        task_id="refresh_marts",
        python_callable=task_refresh_marts,
    )

    ingest_task >> etl_task >> mart_task
