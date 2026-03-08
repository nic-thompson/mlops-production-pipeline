import json
import joblib
import hashlib
import logging
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = "data/train.parquet"

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

# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    ) 

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

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

def save_artifacts(model, metrics, version: str, df: pd.DataFrame):

    artifact_dir = Path("artifacts/models") / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving artifacts to {artifact_dir}")

    # save model
    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path)

    # save metrics
    metrics_path = artifact_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save metadata
    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": version.split("_")[1],
        "model_type": "RandomForrestClassifier",

        "dataset": {
            "path": TRAIN_DATA_PATH,
            "rows": len(df),
            "sha265": dataset_hash("data/train.parquet")
        }
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
    logger.info("Loading training dataset...")

    df = pd.read_parquet(TRAIN_DATA_PATH)

    logger.info("Training model...")

    model, metrics = train_model(df)

    logger.info(f"Training complete. Metrics: {metrics}")

    version = generate_model_version()

    logger.info(f"Generate model version: {version}")

    save_artifacts(model, metrics, version, df)

    logger.info("Training pipeline finished successfully.")

# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()