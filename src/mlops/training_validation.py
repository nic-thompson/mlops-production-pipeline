"""
Training validation utilities for model pipelines.
"""
import json
import logging
from training.train import dataset_hash

# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

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
