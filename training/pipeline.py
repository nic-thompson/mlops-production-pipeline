import json
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
# Production accuracy
# ------------------------

def get_production_accuracy(registry, model_dir):
    """
    Return accuracy of current production model.
    Return None if no production model exists. 
    """

    production_version = registry.get_production()

    if production_version is None:
        return None
    
    metrics_path = model_dir / production_version / "metrics.json"

    if not metrics_path.exists():
        raise RuntimeError(
            f"Production model metrics missing: {metrics_path}"
        )

    with open(metrics_path) as f:
        metrics = json.load(f)

    return metrics["accuracy"]

def validate_candidate(metrics, registry, models_dir, config):
    accuracy = metrics["accuracy"]
    min_accuracy = config["training"]["min_accuracy"]
    margin = config["training"]["improvement_margin"]

    logger.info(
        "Evaluating model: accuracy=%s threshold=%s", 
        accuracy, 
        min_accuracy
    )

    if accuracy < min_accuracy:
        raise RuntimeError(
            f"Model failed evaluation gate: accuracy={accuracy} < {min_accuracy}"
        )
    
    production_accuracy = get_production_accuracy(registry, models_dir)

    if production_accuracy is None:
        return
    
    logger.info(
        "Comparing candidate vs production: candidate=%s production=%s margin=%s",
        accuracy,
        production_accuracy,
        margin
    )

    if accuracy < production_accuracy + margin:
        raise RuntimeError(
            "Candidate model does not sufficiently outperform production."
        )
    
def build_dataset_metadata(df, dataset_path, target_col):
    schema = {c: str(t) for c, t in df.dtypes.items() if c != target_col}

    return {
        "path": dataset_path,
        "rows": len(df),
        "sha256": dataset_hash(dataset_path),
        "columns": [c for c in df.columns if c != target_col],
        "target": target_col,
        "schema": schema
    }

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