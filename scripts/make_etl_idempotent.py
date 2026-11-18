"""
make_etl_idempotent.py
----------------------
Patches etl.py in three places:

1. Before a file's rejected rows are written, any rows already logged for that same file
   are deleted, so reprocessing a file replaces its rejects instead of adding duplicates.
2. A row that fails both the quantity and the price check is logged once, not twice.
3. Unit prices below 0.005 are rejected as invalid_price. They would be stored as 0.00 by the
   NUMERIC(10,2) column, which breaks the rule that every price is above zero.

A backup is saved as etl.py.bak first. Each change is applied only if the exact original
text is found, and the script reports what it did. It is safe to run more than once.

Run from the project root:
    /opt/anaconda3/bin/python scripts/make_etl_idempotent.py
"""
import shutil
from pathlib import Path

ETL_PATH = Path(__file__).resolve().parent.parent / "etl.py"

CHANGES = [
    (
        "delete earlier rejects for the same file",
        "    if rejects:\n        reject_df = pd.concat(",
        "    # Make reruns idempotent: replace this file's earlier rejects instead of adding to them\n"
        "    with engine.begin() as conn:\n"
        "        conn.execute(text(\"DELETE FROM etl_error_log WHERE source_file = :f\"), {\"f\": source_file})\n"
        "\n"
        "    if rejects:\n        reject_df = pd.concat(",
        "DELETE FROM etl_error_log WHERE source_file",
    ),
    (
        "log rows failing both checks only once",
        "rejects.append(df[bad_price].assign(reason=\"invalid_price\"))",
        "rejects.append(df[bad_price & ~bad_qty].assign(reason=\"invalid_price\"))",
        "bad_price & ~bad_qty",
    ),
    (
        "reject prices that would be stored as 0.00",
        "bad_price = df[\"unit_price\"].isna() | (df[\"unit_price\"] <= 0)",
        "bad_price = df[\"unit_price\"].isna() | (df[\"unit_price\"] < 0.005)",
        "< 0.005",
    ),
]


def main():
    if not ETL_PATH.exists():
        print(f"etl.py not found at {ETL_PATH}. Run this from the project root.")
        return
    source = ETL_PATH.read_text()
    updated = source

    for label, old, new, marker in CHANGES:
        if marker in updated:
            print(f"already applied: {label}")
        elif old in updated:
            updated = updated.replace(old, new, 1)
            print(f"applied: {label}")
        else:
            print(f"NOT FOUND: {label}. Your etl.py differs from the expected text, so this "
                  f"change was skipped. Send me the clean() function and I will adjust it.")

    if updated != source:
        shutil.copy(ETL_PATH, ETL_PATH.with_name("etl.py.bak"))
        ETL_PATH.write_text(updated)
        compile(updated, str(ETL_PATH), "exec")   # raises if the patched file is not valid Python
        print("etl.py updated (backup saved as etl.py.bak) and checked for syntax errors.")
    else:
        print("No changes were needed.")


if __name__ == "__main__":
    main()
