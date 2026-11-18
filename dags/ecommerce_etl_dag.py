import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from dotenv import load_dotenv

PROJECT_ROOT = "/Users/keneshawatt/Documents/ecommerce_MLOPS"
PYTHON_BIN = "/opt/anaconda3/bin/python"

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
DATABASE_URL = os.getenv("DATABASE_URL")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ecommerce_monthly_etl",
    default_args=default_args,
    schedule="@monthly",
    catchup=False,
) as dag:

    landing_dir = os.path.join(PROJECT_ROOT, "data/landing")
    files = sorted([f for f in os.listdir(landing_dir) if f.endswith(".csv")])

    previous_task = None
    for file_name in files:
        task_id = f"ingest_{file_name.replace('.csv', '').replace('-', '_')}"
        
        task = BashOperator(
            task_id=task_id,
            bash_command=(
                f"cd {PROJECT_ROOT} && "
                f"export DATABASE_URL='{DATABASE_URL}' && "
                f"{PYTHON_BIN} etl.py --file data/landing/{file_name}"
            ),
        )

        if previous_task:
            previous_task >> task
        previous_task = task