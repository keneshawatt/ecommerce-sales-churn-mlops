"""
export_sample_tables.py
-----------------------
Saves the first rows of every warehouse table and mart to CSV files in
docs/sample_tables/ (the "database schema and sample populated tables" item of the
submission package), plus an INDEX.md with full row counts.

Run from the project root:
    /opt/anaconda3/bin/python scripts/export_sample_tables.py
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
OUT_DIR = PROJECT_ROOT / "docs" / "sample_tables"
SAMPLE_ROWS = 20

SAMPLES = {
    "dim_customer": "SELECT * FROM dim_customer ORDER BY 1 LIMIT {n}",
    "dim_product": "SELECT * FROM dim_product ORDER BY 1 LIMIT {n}",
    "dim_geography": "SELECT * FROM dim_geography ORDER BY 1 LIMIT {n}",
    "dim_date": "SELECT * FROM dim_date ORDER BY 1 LIMIT {n}",
    "fact_sales": "SELECT * FROM fact_sales ORDER BY 1 LIMIT {n}",
    "fact_customer_retention": "SELECT * FROM fact_customer_retention ORDER BY 1 LIMIT {n}",
    "etl_run_log": "SELECT * FROM etl_run_log ORDER BY 1 LIMIT {n}",
    "etl_error_log": "SELECT * FROM etl_error_log ORDER BY 1 LIMIT {n}",
    "sales_mart": "SELECT * FROM sales_mart ORDER BY total_revenue DESC LIMIT {n}",
    "customer_mart": "SELECT * FROM customer_mart ORDER BY monetary DESC LIMIT {n}",
}


def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = ["# Sample populated tables", "",
             f"First {SAMPLE_ROWS} rows of each table (marts are sorted by revenue, highest first).", "",
             "| Table | Rows in database | Sample file |", "|---|---|---|"]

    with engine.connect() as conn:
        for name, query in SAMPLES.items():
            try:
                df = pd.read_sql(query.format(n=SAMPLE_ROWS), conn)
                total = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            except Exception as exc:
                conn.rollback()
                print(f"skipped {name}: {type(exc).__name__}")
                continue
            if "row_data" in df.columns:
                df["row_data"] = df["row_data"].astype(str).str.slice(0, 150)
            df.to_csv(OUT_DIR / f"{name}.csv", index=False)
            index.append(f"| {name} | {total:,} | {name}.csv |")
            print(f"saved {name}: {len(df)} sample rows ({total:,} in the table)")

    (OUT_DIR / "INDEX.md").write_text("\n".join(index) + "\n")
    print(f"\nSamples written to {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
