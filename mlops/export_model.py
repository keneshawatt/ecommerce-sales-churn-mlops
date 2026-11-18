"""
export_model.py
---------------
Exports the latest registered 'churn-model' from the local MLflow registry into
a plain folder (models/churn-model). A Docker container cannot see your laptop's
mlflow.db, so the container loads this folder instead. MLflow also writes a
pinned requirements.txt inside the folder, which the Dockerfile uses so the
container has the same library versions the model was trained with.

Run from the project root, after train.py:
    /opt/anaconda3/bin/python mlops/export_model.py
"""
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "churn-model"
OUT_DIR = PROJECT_ROOT / "models" / "churn-model"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(v.version) for v in versions)
    print(f"Exporting {MODEL_NAME} version {latest} ...")

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.parent.mkdir(exist_ok=True)
    mlflow.sklearn.save_model(model, str(OUT_DIR), serialization_format="cloudpickle")
    (OUT_DIR / "model_version.txt").write_text(str(latest))

    print(f"Saved to {OUT_DIR}")
    print("Files:", sorted(p.name for p in OUT_DIR.iterdir()))


if __name__ == "__main__":
    main()
