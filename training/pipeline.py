import json
import yaml
import logging
from pathlib import Path
import pandas as pd

from src.mlops.training_validation import (
    validate_candidate,
    build_dataset_metadata,
)

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

    models_dir = Path(config["output"]["model_dir"])
    registry = ModelRegistry(Path(models_dir / "registry.json"))

    validate_candidate(metrics, registry, models_dir, config)

    version = generate_model_version()

    logger.info(f"Generated model version: {version}")

    dataset_metadata = build_dataset_metadata(df, dataset_path, target_col)

    save_artifacts(
        model=model, 
        metrics=metrics, 
        version=version, 
        dataset_metadata=dataset_metadata
    )

    registry.promote_to_staging(version)

    logger.info(f"Model {version} promoted to staging.")
    logger.info("Pipeline completed successfully.")

    return version

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    run_pipeline()