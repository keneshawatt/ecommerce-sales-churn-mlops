"""
build_marts_and_retention.py
----------------------------
1. (Re)creates the sales_mart and customer_mart views from sql/marts.sql.
2. Creates (if needed) and fills the fact_customer_retention fact table: one row per
   customer, as of the last transaction date in the warehouse, with recency,
   frequency, monetary value and a repeat-customer flag.

Safe to run repeatedly: the views are replaced and the fact rows are upserted.

Run from the project root:
    /opt/anaconda3/bin/python scripts/build_marts_and_retention.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CREATE_RETENTION = """
CREATE TABLE IF NOT EXISTS fact_customer_retention (
    retention_key      BIGSERIAL PRIMARY KEY,
    customer_key       INTEGER REFERENCES dim_customer(customer_key),
    snapshot_date      DATE NOT NULL,
    recency_days       INTEGER,
    frequency          INTEGER,
    monetary           NUMERIC(12,2),
    is_repeat_customer BOOLEAN,
    UNIQUE (customer_key, snapshot_date)
)
"""

LOAD_RETENTION = """
INSERT INTO fact_customer_retention
    (customer_key, snapshot_date, recency_days, frequency, monetary, is_repeat_customer)
SELECT
    f.customer_key,
    a.as_of,
    (a.as_of - MAX(d.full_date)),
    COUNT(DISTINCT f.invoice_no),
    SUM(f.revenue),
    COUNT(DISTINCT f.invoice_no) > 1
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
CROSS JOIN (SELECT MAX(full_date) AS as_of FROM dim_date) a
GROUP BY f.customer_key, a.as_of
ON CONFLICT (customer_key, snapshot_date) DO UPDATE SET
    recency_days       = EXCLUDED.recency_days,
    frequency          = EXCLUDED.frequency,
    monetary           = EXCLUDED.monetary,
    is_repeat_customer = EXCLUDED.is_repeat_customer
"""


def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    marts_sql = (PROJECT_ROOT / "sql" / "marts.sql").read_text()

    with engine.begin() as conn:
        conn.execute(text(marts_sql))
        print("Marts created: sales_mart, customer_mart")
        conn.execute(text(CREATE_RETENTION))
        conn.execute(text(LOAD_RETENTION))
        rows = conn.execute(text("SELECT COUNT(*) FROM fact_customer_retention")).scalar()
        repeat = conn.execute(text(
            "SELECT COUNT(*) FROM fact_customer_retention WHERE is_repeat_customer")).scalar()
    print(f"fact_customer_retention: {rows} customers, {repeat} repeat customers")


if __name__ == "__main__":
    main()
