"""
etl.py
------
Optimized Star Schema ETL pipeline for Neon PostgreSQL.
"""
import argparse
import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

RUN_ID_PREFIX = "etl"

def get_engine():
    db_url = os.environ.get(
        "DATABASE_URL"
    )
    return create_engine(db_url, connect_args={"connect_timeout": 10})

def create_schema_if_not_exists(engine):
    """Ensure destination tables exist prior to insertion."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dim_customer (
                customer_key SERIAL PRIMARY KEY,
                customer_id VARCHAR(50) UNIQUE NOT NULL,
                country VARCHAR(100),
                first_purchase_date TIMESTAMP,
                updated_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS dim_product (
                product_key SERIAL PRIMARY KEY,
                stock_code VARCHAR(50) UNIQUE NOT NULL,
                description TEXT
            );
            CREATE TABLE IF NOT EXISTS dim_geography (
                geography_key SERIAL PRIMARY KEY,
                country VARCHAR(100) UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dim_date (
                date_key INT PRIMARY KEY,
                full_date DATE,
                day INT, month INT, month_name VARCHAR(20),
                quarter INT, year INT, day_of_week VARCHAR(20), is_weekend BOOLEAN
            );
            CREATE TABLE IF NOT EXISTS fact_sales (
                fact_id SERIAL PRIMARY KEY,
                invoice_no VARCHAR(50),
                date_key INT,
                customer_key INT,
                product_key INT,
                geography_key INT,
                quantity INT,
                unit_price NUMERIC(10,2),
                revenue NUMERIC(12,2),
                CONSTRAINT unique_invoice_product UNIQUE (invoice_no, product_key)
            );
            CREATE TABLE IF NOT EXISTS etl_error_log (
                log_id SERIAL PRIMARY KEY,
                run_id VARCHAR(50),
                source_file VARCHAR(255),
                error_reason VARCHAR(100),
                row_data TEXT
            );
            CREATE TABLE IF NOT EXISTS etl_run_log (
                log_id SERIAL PRIMARY KEY,
                run_id VARCHAR(50),
                source_name VARCHAR(255),
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                status VARCHAR(50),
                rows_extracted INT,
                rows_loaded INT,
                rows_rejected INT
            );
        """))

def clean(df: pd.DataFrame, run_id: str, engine, source_file: str):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "invoiceno": "invoice_no",
        "stockcode": "stock_code",
        "customerid": "customer_id",
        "unitprice": "unit_price",
        "invoicedate": "invoice_date",
    }
    df = df.rename(columns=rename_map)

    initial_rows = len(df)
    rejects = []

    is_cancelled = df["invoice_no"].astype(str).str.startswith("C")
    rejects.append(df[is_cancelled].assign(reason="cancelled_order"))
    df = df[~is_cancelled]

    bad_qty = df["quantity"].isna() | (df["quantity"] <= 0)
    bad_price = df["unit_price"].isna() | (df["unit_price"] < 0.005)
    rejects.append(df[bad_qty].assign(reason="invalid_quantity"))
    rejects.append(df[bad_price & ~bad_qty].assign(reason="invalid_price"))
    df = df[~bad_qty & ~bad_price]

    bad_cust = df["customer_id"].isna()
    rejects.append(df[bad_cust].assign(reason="missing_customer_id"))
    df = df[~bad_cust]

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["customer_id"] = df["customer_id"].astype(str).str.replace(".0", "", regex=False)
    df["country"] = df["country"].astype(str).str.strip().str.title()

    # Make reruns idempotent: replace this file's earlier rejects instead of adding to them
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM etl_error_log WHERE source_file = :f"), {"f": source_file})

    if rejects:
        reject_df = pd.concat([r for r in rejects if len(r)], ignore_index=True) if any(len(r) for r in rejects) else pd.DataFrame()
        if len(reject_df):
            reject_df["run_id"] = run_id
            reject_df["source_file"] = source_file
            log_rows = reject_df[["run_id", "source_file", "reason"]].rename(columns={"reason": "error_reason"})
            log_rows["row_data"] = reject_df.astype(str).to_dict("records")
            log_rows["row_data"] = log_rows["row_data"].astype(str)
            log_rows[["run_id", "source_file", "row_data", "error_reason"]].to_sql(
                "etl_error_log", engine, if_exists="append", index=False, chunksize=1000, method="multi"
            )

    rejected_count = initial_rows - len(df)
    return df, initial_rows, rejected_count

def upsert_dimensions(df: pd.DataFrame, engine):
    with engine.begin() as conn:
        # Batch insert dim_customer
        customers = df[["customer_id", "country"]].drop_duplicates("customer_id")
        customers.to_sql("temp_cust", conn, if_exists="replace", index=False)
        conn.execute(text("""
            INSERT INTO dim_customer (customer_id, country, first_purchase_date)
            SELECT customer_id, country, NOW() FROM temp_cust
            ON CONFLICT (customer_id) DO UPDATE SET updated_at = NOW();
            DROP TABLE IF EXISTS temp_cust;
        """))

        # Batch insert dim_product
        products = df[["stock_code", "description"]].drop_duplicates("stock_code")
        products["description"] = products["description"].astype(str).str.slice(0, 500)
        products.to_sql("temp_prod", conn, if_exists="replace", index=False)
        conn.execute(text("""
            INSERT INTO dim_product (stock_code, description)
            SELECT stock_code, description FROM temp_prod
            ON CONFLICT (stock_code) DO NOTHING;
            DROP TABLE IF EXISTS temp_prod;
        """))

        # Batch insert dim_geography
        geos = pd.DataFrame({"country": df["country"].dropna().unique()})
        geos.to_sql("temp_geo", conn, if_exists="replace", index=False)
        conn.execute(text("""
            INSERT INTO dim_geography (country)
            SELECT country FROM temp_geo
            ON CONFLICT (country) DO NOTHING;
            DROP TABLE IF EXISTS temp_geo;
        """))

        # Batch insert dim_date
        dates = pd.DataFrame({"d": df["invoice_date"].dropna().dt.date.unique()})
        if not dates.empty:
            dates["date_key"] = dates["d"].apply(lambda x: int(pd.Timestamp(x).strftime("%Y%m%d")))
            dates["full_date"] = dates["d"]
            dates["day"] = dates["d"].apply(lambda x: x.day)
            dates["month"] = dates["d"].apply(lambda x: x.month)
            dates["month_name"] = dates["d"].apply(lambda x: pd.Timestamp(x).strftime("%B"))
            dates["quarter"] = dates["d"].apply(lambda x: (x.month - 1) // 3 + 1)
            dates["year"] = dates["d"].apply(lambda x: x.year)
            dates["day_of_week"] = dates["d"].apply(lambda x: pd.Timestamp(x).strftime("%A"))
            dates["is_weekend"] = dates["d"].apply(lambda x: x.weekday() >= 5)
            
            dates_to_load = dates.drop(columns=["d"])
            dates_to_load.to_sql("temp_date", conn, if_exists="replace", index=False)
            conn.execute(text("""
                INSERT INTO dim_date (date_key, full_date, day, month, month_name, quarter, year, day_of_week, is_weekend)
                SELECT date_key, full_date, day, month, month_name, quarter, year, day_of_week, is_weekend FROM temp_date
                ON CONFLICT (date_key) DO NOTHING;
                DROP TABLE IF EXISTS temp_date;
            """))

def load_fact_sales(df: pd.DataFrame, engine):
    df = df.copy()
    df["date_key"] = df["invoice_date"].dt.strftime("%Y%m%d").astype(int)
    df["revenue"] = df["quantity"] * df["unit_price"]

    dims = pd.read_sql("SELECT customer_key, customer_id FROM dim_customer", engine)
    prods = pd.read_sql("SELECT product_key, stock_code FROM dim_product", engine)
    geo = pd.read_sql("SELECT geography_key, country FROM dim_geography", engine)

    df = df.merge(dims, on="customer_id", how="left")
    df = df.merge(prods, on="stock_code", how="left")
    df = df.merge(geo, on="country", how="left")

    fact = df[["invoice_no", "date_key", "customer_key", "product_key",
               "geography_key", "quantity", "unit_price", "revenue"]].drop_duplicates(["invoice_no", "product_key"])

    with engine.begin() as conn:
        fact.to_sql("temp_fact", conn, if_exists="replace", index=False, chunksize=1000, method="multi")
        conn.execute(text("""
            INSERT INTO fact_sales (invoice_no, date_key, customer_key, product_key, geography_key, quantity, unit_price, revenue)
            SELECT invoice_no, date_key, customer_key, product_key, geography_key, quantity, unit_price, revenue FROM temp_fact
            ON CONFLICT (invoice_no, product_key) DO NOTHING;
            DROP TABLE IF EXISTS temp_fact;
        """))

    return len(fact)

def run_etl(file_path: str):
    engine = get_engine()
    create_schema_if_not_exists(engine)
    
    run_id = f"{RUN_ID_PREFIX}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    started = datetime.now()

    raw = pd.read_csv(file_path, encoding="ISO-8859-1")
    cleaned, initial_rows, rejected = clean(raw, run_id, engine, os.path.basename(file_path))
    upsert_dimensions(cleaned, engine)
    loaded = load_fact_sales(cleaned, engine)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO etl_run_log (run_id, source_name, started_at, finished_at, status, rows_extracted, rows_loaded, rows_rejected)
            VALUES (:rid, :src, :st, :fn, :status, :ext, :ld, :rej)
        """), {
            "rid": run_id, "src": os.path.basename(file_path), "st": started,
            "fn": datetime.now(), "status": "SUCCESS", "ext": initial_rows,
            "ld": loaded, "rej": rejected,
        })

    print(f"[{run_id}] extracted={initial_rows} loaded={loaded} rejected={rejected}")
    
    # Safely close database connections
    engine.dispose()
    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Landing CSV file to process")
    args = parser.parse_args()
    run_etl(args.file)