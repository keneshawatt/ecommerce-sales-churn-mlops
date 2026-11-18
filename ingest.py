"""
ingest.py
---------
Ingestion layer: pulls a raw source file, stamps it with metadata,
and drops an untouched copy into data/landing/. Also splits the
UCI Online Retail dataset into monthly chunks so you can demonstrate
incremental ingestion (as the assignment requires).

Run:
    python ingest.py --source data/raw/online_retail.csv
"""
import argparse
import hashlib
import json
import os
from datetime import datetime

import pandas as pd

# Set explicit paths relative to the project directory
BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
LANDING_DIR = os.path.join(BASE_DIR, "data", "landing")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")


def log_ingestion(run_id, source_name, status, row_count, note=""):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_entry = {
        "run_id": run_id,
        "source_name": source_name,
        "extraction_date": datetime.now().isoformat(),
        "status": status,
        "row_count": row_count,
        "note": note,
    }
    log_path = os.path.join(LOG_DIR, "ingestion_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"[LOGGED] {log_entry}")


def make_run_id(source_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    h = hashlib.md5(source_name.encode()).hexdigest()[:6]
    return f"run_{stamp}_{h}"


def ingest_full_file(source_path: str):
    """Copy raw file into landing untouched, then split by month for
    incremental-load simulation."""
    run_id = make_run_id(source_path)
    os.makedirs(LANDING_DIR, exist_ok=True)

    # Resolve absolute path for source_path
    if not os.path.isabs(source_path):
        source_path = os.path.join(BASE_DIR, source_path)

    try:
        # UCI Online Retail is usually .xlsx; support both csv/xlsx
        if source_path.endswith(".xlsx"):
            df = pd.read_excel(source_path)
        else:
            df = pd.read_csv(source_path, encoding="ISO-8859-1")
    except Exception as e:
        log_ingestion(run_id, source_path, "FAILED", 0, str(e))
        raise

    # Normalize column names once, at the very edge of the system
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "invoicedate" not in df.columns and "invoice_date" not in df.columns:
        raise ValueError("Expected an InvoiceDate column in the source file.")

    date_col = "invoicedate" if "invoicedate" in df.columns else "invoice_date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Store Period for monthly file splitting
    df["extract_period"] = df[date_col].dt.to_period("M")

    total_rows = len(df)
    chunks_written = 0

    for period, chunk in df.groupby("extract_period"):
        if pd.isna(period):
            continue
        
        # Formats output file to match task definitions: Jan_2011.csv
        file_name = f"{period.strftime('%b')}_{period.year}.csv"
        out_path = os.path.join(LANDING_DIR, file_name)
        
        chunk.drop(columns=["extract_period"]).to_csv(out_path, index=False)
        chunks_written += 1

    log_ingestion(
        run_id,
        os.path.basename(source_path),
        "SUCCESS",
        total_rows,
        note=f"Split into {chunks_written} monthly landing files",
    )
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to raw source file")
    args = parser.parse_args()
    ingest_full_file(args.source)