import logging
from pathlib import Path

from src.registry import ModelRegistry
from training.train import generate_model_version, train_model, save_artifacts

import pandas as pd

MIN_ACCURACY = 0.85

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

    logger.info("Loading training dataset...")
    df = pd.read_parquet("data/train.parquet")

    logger.info("Training model...")
    model, metrics = train_model(df)

    logger.info(f"Training complete. Metrics: {metrics}")

    # Evaluation Gate
    accuracy = metrics["accuracy"]

    if accuracy < MIN_ACCURACY:
        logger.error(
            f"Model failed evaluation gate: accuracy={accuracy} < threshold={MIN_ACCURACY}"
        )
        raise RuntimeError("Model did not meet minimum accuracy threshold.")

    version = generate_model_version()

    logger.info(f"Generated model version: {version}")

    artifact_dir = save_artifacts(model, metrics, version)

    logger.info("Initialising model registry...")

    registry = ModelRegistry(Path("artifacts/models/registry.json"))

    logger.info("Registering model in staging...")

    registry.promote_to_staging(version)

    logger.info(f"Model {version} promoted to staging.")

    logger.info("Pipeline completed successfully.")

    return version

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    run_pipeline()