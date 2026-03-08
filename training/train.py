import json
import yaml
import joblib
import hashlib
import logging
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.base import BaseEstimator
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def get_git_commit() -> str:
    """Return the short git commit hash."""
    try: 
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"
    
def generate_model_version() -> str:
    """Create version string based on timestamp + git commit."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    git_commit = get_git_commit()
    return f"{timestamp}_{git_commit}"

def dataset_hash(path: str) -> str:
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)

    return sha.hexdigest()

def load_config(path="config/training.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(
        df: pd.DataFrame,
        target_col: str,
        test_size: float,
        random_state: int,
        model_params: Dict    
    ) -> Tuple[BaseEstimator, Dict]:

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    ) 

    model = RandomForestClassifier(**model_params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "accuracy": accuracy
    }

    return model, metrics

# --------------------------------------------------
# Atrtifact Writing
# --------------------------------------------------

def save_artifacts(
    model, 
    metrics: dict, 
    version: str,
    dataset_metadata: dict
):

    artifact_dir = Path("artifacts/models") / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving artifacts to {artifact_dir}")

    # --------------------------
    # Save model
    # --------------------------

    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path)

    # --------------------------
    # Save metricts
    # --------------------------

    metrics_path = artifact_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --------------------------
    # Save metadata
    # --------------------------

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": version.split("_")[1],
        "model_type": model.__class__.__name__,

        "dataset": dataset_metadata,

        "hyperparameters": model.get_params()
    }

    metadata_path = artifact_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts saved successfully")

    return artifact_dir

# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():

    config = load_config()

    dataset_path = config["dataset"]["path"]
    target_col = config["training"]["target_column"]

    logger.info("Loading training dataset...")
    df = pd.read_parquet(dataset_path)

    model, metrics = train_model(
        df,
        target_col,
        config["dataset"]["test_size"],
        config["dataset"]["random_state"],
        config["model"]["params"]
    )

    version = generate_model_version()

    logger.info(f"Generate model version: {version}")

    schema = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
         if col != target_col
    }

    dataset_metadata = {
        "path": dataset_path,
        "rows": len(df),
        "sha256": dataset_hash(dataset_path),
        "columns": [c for c in df.columns if c != target_col],
        "target": "target",
        "schema": schema
    }

    save_artifacts(
        model=model, 
        metrics=metrics, 
        version=version,
        dataset_metadata=dataset_metadata)

    logger.info("Training pipeline finished successfully.")

# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()