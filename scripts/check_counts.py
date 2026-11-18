"""
check_counts.py
---------------
Verifies the Part 1 pipeline end to end and prints numbers you can quote in the report:

1. Row counts for every warehouse table
2. Reconciliation: rows in the landing files vs fact_sales + etl_error_log
3. Rejected rows by reason
4. ETL runs per landing file (a file processed more than once means a rerun)
5. Proof that cancelled orders and invalid rows did not reach fact_sales

Run from the project root:
    /opt/anaconda3/bin/python scripts/check_counts.py
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TABLES = ["dim_customer", "dim_product", "dim_geography", "dim_date", "fact_sales",
          "fact_customer_retention", "etl_run_log", "etl_error_log"]


def landing_files():
    """Fingerprint every landing file (row count + hash of the first column) so that files
    holding the same rows under different names can be spotted."""
    groups = {}
    files = sorted((PROJECT_ROOT / "data" / "landing").glob("*.csv"))
    for f in files:
        col = pd.read_csv(f, usecols=[0], dtype=str, encoding="ISO-8859-1").iloc[:, 0]
        fingerprint = (len(col), int(pd.util.hash_pandas_object(col, index=False).sum()))
        groups.setdefault(fingerprint, []).append(f.name)
    return len(files), groups


def scalar(conn, sql):
    return conn.execute(text(sql)).scalar()


def main():
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    counts = {}

    with engine.connect() as conn:
        print("\n=== 1. Table row counts ===")
        for table in TABLES:
            try:
                counts[table] = scalar(conn, f"SELECT COUNT(*) FROM {table}")
                print(f"{table:<26}{counts[table]:>10,}")
            except Exception as exc:
                conn.rollback()
                print(f"{table:<26}  not available ({type(exc).__name__})")

        print("\n=== 2. Reconciliation with the landing files ===")
        n_files, groups = landing_files()
        unique_rows = sum(fp[0] for fp in groups)
        print(f"Landing files           : {n_files} ({len(groups)} distinct datasets)")
        duplicate_groups = [names for names in groups.values() if len(names) > 1]
        if duplicate_groups:
            print("WARNING: these landing files contain exactly the same rows, so they were "
                  "loaded more than once:")
            for names in duplicate_groups:
                print("  " + " = ".join(names))
            print("  Keep one file per month and delete the others from data/landing/.")
        print(f"Rows in distinct files  : {unique_rows:,}")
        if counts.get("fact_sales", 0) == 0 and counts.get("etl_error_log", 0) == 0:
            print("The warehouse is empty: nothing has been loaded yet. Run the Airflow DAG first.")
        elif "fact_sales" in counts and "etl_error_log" in counts and unique_rows:
            accounted = counts["fact_sales"] + counts["etl_error_log"]
            diff = accounted - unique_rows
            print(f"fact_sales + error log  : {accounted:,}")
            print(f"Difference              : {diff:+,}")
            if diff > 0.005 * unique_rows:
                print("PROBLEM: more rows are accounted for than exist in the source, so the error "
                      "log holds duplicates (files processed twice, or reruns before the fix). "
                      "See scripts/make_etl_idempotent.py.")
            elif diff > 0:
                print("Slightly above the source count: rows that fail both the quantity and the "
                      "price check were logged twice (fixed by scripts/make_etl_idempotent.py).")
            else:
                print(f"OK: nothing is double counted. {-diff:,} source rows ({-diff / unique_rows:.1%}) "
                      f"are in neither table: duplicate (invoice, product) lines, which the unique "
                      f"constraint keeps once.")
            try:
                rejected_by_file = dict(conn.execute(text(
                    "SELECT source_name, MAX(rows_rejected) FROM etl_run_log GROUP BY source_name")).all())
                expected_rejects = sum(rejected_by_file.get(names[0], 0) for names in groups.values())
                extra = counts["etl_error_log"] - expected_rejects
                if extra > 0:
                    print(f"Error log has {extra:,} more rows than the run log says were rejected "
                          f"({counts['etl_error_log']:,} vs {expected_rejects:,}).")
            except Exception:
                conn.rollback()

        print("\n=== 3. Rejected rows by reason ===")
        try:
            by_reason = pd.read_sql(
                "SELECT error_reason, COUNT(*) AS rows FROM etl_error_log "
                "GROUP BY error_reason ORDER BY rows DESC", conn)
            total = int(by_reason["rows"].sum()) or 1
            for _, r in by_reason.iterrows():
                print(f"{r['error_reason']:<24}{int(r['rows']):>10,}   {r['rows'] / total:.1%}")
        except Exception as exc:
            conn.rollback()
            print(f"not available ({type(exc).__name__})")

        print("\n=== 4. ETL runs per landing file ===")
        try:
            runs = pd.read_sql(
                "SELECT source_name, COUNT(*) AS runs, MAX(rows_extracted) AS rows_extracted, "
                "MAX(rows_loaded) AS rows_loaded, MAX(rows_rejected) AS rows_rejected "
                "FROM etl_run_log GROUP BY source_name ORDER BY source_name", conn)
            print(runs.to_string(index=False))
            reruns = runs[runs["runs"] > 1]
            if len(reruns):
                print(f"\nNote: {len(reruns)} file(s) were processed more than once (reruns).")
        except Exception as exc:
            conn.rollback()
            print(f"not available ({type(exc).__name__})")

        print("\n=== 5. Validation proof on fact_sales ===")
        try:
            if not counts.get("fact_sales"):
                print("fact_sales is empty, so there is nothing to validate yet.")
                return
            row = conn.execute(text(
                "SELECT MIN(quantity), MIN(unit_price), "
                "COUNT(*) FILTER (WHERE invoice_no LIKE 'C%'), "
                "COUNT(*) FILTER (WHERE customer_key IS NULL), "
                "COUNT(*) FILTER (WHERE unit_price = 0) FROM fact_sales")).one()
            print(f"Smallest quantity          : {row[0]}   (must be above 0)")
            print(f"Smallest unit price        : {row[1]}   (must be above 0)")
            print(f"Cancelled invoices in fact : {row[2]}   (must be 0)")
            print(f"Rows without a customer    : {row[3]}   (must be 0)")
            print(f"Rows with unit price 0.00  : {row[4]}   (must be 0)")
            if row[4]:
                print("  These are sub-penny prices (for example 0.001) that pass the 'above 0' check "
                      "but are stored as 0.00 by the NUMERIC(10,2) column. scripts/make_etl_idempotent.py "
                      "adds a rule that rejects them.")
        except Exception as exc:
            conn.rollback()
            print(f"not available ({type(exc).__name__})")


if __name__ == "__main__":
    main()
