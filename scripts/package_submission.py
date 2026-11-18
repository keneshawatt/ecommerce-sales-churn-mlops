"""
package_submission.py
---------------------
Builds submission/<project>_submission.zip with the source code, DAG, SQL, docs, sample
tables, screenshots and README, and leaves out everything that must not be submitted:
.env, mlflow.db, mlruns/, the raw and landing data, trained model files, caches.

Before zipping it scans every text file for database URLs that contain a real password
and for Neon-style password strings. If it finds one it stops and tells you which file and
line to fix. It never prints the secret itself.

Run from the project root:
    /opt/anaconda3/bin/python scripts/package_submission.py
"""
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "submission"

INCLUDE_DIRS = ["dags", "dashboard", "docs", "mlops", "scripts", "sql", "screenshots", "monitoring"]
INCLUDE_FILES = ["ingest.py", "etl.py", "README.md", "docker-compose.yml", "Dockerfile.api",
                 "Dockerfile.dashboard", ".dockerignore", ".gitignore", "requirements.txt",
                 "data/logs/ingestion_log.jsonl"]
SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", "venv", "mlruns", "node_modules", ".ipynb_checkpoints"}
SKIP_FILE_NAMES = {".env", "mlflow.db", ".DS_Store", "etl.py.bak"}
TEXT_SUFFIXES = {".py", ".sql", ".md", ".yml", ".yaml", ".txt", ".json", ".jsonl", ".csv",
                 ".cfg", ".toml", ".dockerignore", ".gitignore", ""}
MAX_SCAN_BYTES = 5_000_000

DB_URL = re.compile(r"postgres(?:ql)?(?:\+\w+)?://([^:/\s'\"@]+):([^@\s'\"]+)@")
NEON_PASSWORD = re.compile(r"npg_[A-Za-z0-9]{8,}")
PLACEHOLDERS = {"password", "your_password", "new_password_here", "***", "pass", "xxxx"}


def collect_files():
    files = []
    for name in INCLUDE_FILES:
        path = PROJECT_ROOT / name
        if path.is_file():
            files.append(path)
    for folder in INCLUDE_DIRS:
        base = PROJECT_ROOT / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if SKIP_DIR_NAMES & set(path.relative_to(PROJECT_ROOT).parts):
                continue
            if path.name in SKIP_FILE_NAMES or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(set(files))


def find_secrets(files):
    findings = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > MAX_SCAN_BYTES:
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            if NEON_PASSWORD.search(line):
                findings.append((path, number, "Neon-style password string"))
                continue
            for match in DB_URL.finditer(line):
                password = match.group(2).strip().lower()
                if password not in PLACEHOLDERS and not password.startswith("<"):
                    findings.append((path, number, "database URL with a password"))
    return findings


def main():
    files = collect_files()
    print(f"Found {len(files)} files to include.")

    findings = find_secrets(files)
    if findings:
        print("\nSTOPPED: possible secrets found. Remove or replace these before packaging:")
        for path, number, kind in findings:
            print(f"  {path.relative_to(PROJECT_ROOT)} line {number}: {kind}")
        sys.exit(1)
    print("Secret scan passed: no credentials found in the included files.")

    OUT_DIR.mkdir(exist_ok=True)
    zip_path = OUT_DIR / f"{PROJECT_ROOT.name}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(PROJECT_ROOT.name) / path.relative_to(PROJECT_ROOT))

    size_mb = zip_path.stat().st_size / 1_000_000
    print(f"\nCreated {zip_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB, {len(files)} files)")
    print("Left out on purpose: .env, mlflow.db, mlruns/, models/, data/raw, data/landing, logs/.")
    for folder in ("screenshots", "docs/sample_tables"):
        if not (PROJECT_ROOT / folder).is_dir():
            print(f"Note: {folder}/ does not exist yet, so it is not in the zip.")


if __name__ == "__main__":
    main()
