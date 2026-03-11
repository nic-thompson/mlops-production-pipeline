import yaml
import logging
from pathlib import Path
import pandas as pd



from src.mlops.registry import ModelRegistry
from training.train import (
    generate_model_version, 
    train_model, 
    save_artifacts,
    dataset_hash
)

# ---------------------------
# Config
# ---------------------------

def load_config(config_path="config/training.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ------------------------
# Pipeline
# ------------------------

def run_pipeline():

    config = load_config()

    dataset_path = config["dataset"]["path"]
    target_col = config["training"]["target_column"]

    logger.info("Loading training dataset...")
    df = pd.read_parquet(dataset_path)


    logger.info("Training model...")

    model, metrics = train_model(
        df,
        target_col,
        config["dataset"]["test_size"],
        config["dataset"]["random_state"],
        config["model"]["params"]
    )

    logger.info(f"Training complete. Metrics: {metrics}")


    # Evaluation Gate
    accuracy = metrics["accuracy"]
    min_accuracy = config["training"]["min_accuracy"]

    logger.info(
        "Evaluating model: accuracy=%s threshold=%s", 
        accuracy, 
        min_accuracy
    )

    if accuracy < min_accuracy:
        logger.error(
            f"Model failed evaluation gate: accuracy={accuracy} < threshold={min_accuracy}"
        )
        raise RuntimeError("Model did not meet minimum accuracy threshold.")

    version = generate_model_version()

    logger.info(f"Generated model version: {version}")

    # Build dataset schema
    schema = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
        if col != target_col
    }

    dataset_metadata = {
        "path": dataset_path,
        "rows": len(df),
        "sha265": dataset_hash(dataset_path),
        "columns": [c for c in df.columns if c != target_col],
        "target": target_col,
        "schema": schema 
    }

    save_artifacts(
        model=model, 
        metrics=metrics, 
        version=version, 
        dataset_metadata=dataset_metadata
    )

    logger.info("Initialising model registry...")

    registry = ModelRegistry(Path("artifacts/models/registry.json"))

    registry.promote_to_staging(version)

    logger.info(f"Model {version} promoted to staging.")

    logger.info("Pipeline completed successfully.")

    return version

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    run_pipeline()