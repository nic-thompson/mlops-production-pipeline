import argparse
import json
import logging
import yaml
import joblib
import pandas as pd

from pathlib import Path

from mlops.registry import ModelRegistry
from mlops.schema_validation import validate_schema
from mlops.prediction_logger import log_predictions

# ----------------------------------------------------
# Logging
# ----------------------------------------------------

def configure_logging(log_level: str):
    log_level = log_level.upper()

    if not hasattr(logging, log_level):
        raise ValueError(f"Invalid log level: {log_level}")
    
    numeric_level = getattr(logging, log_level)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ----------------------------------------------------
# CLI
# ----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Inference pipeline")

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--version",
        default=None,
        help="Explicit model version override (bypasses production)."
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    return parser.parse_args()

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_model_version(registry: ModelRegistry, override: str | None):

    if override:
         if not registry.version_exists(override):
              raise RuntimeError(
                   f"Requested version '{override}' not found in registry."
              )
         return override, "override"
    
    version = registry.get_production()

    if version is None:
         raise RuntimeError("No production model set.")
    
    return version, "production"

def load_model(models_path: Path, version: str):

     model_path = models_path / version / "model.joblib"

     if not model_path.exists():
          raise FileNotFoundError(f"Model artifact missing: {model_path}")
     
     return joblib.load(model_path)

def load_metadata(models_path: Path, version: str):

     metadata_path = models_path / version / "metadata.json"

     if not metadata_path.exists():
          raise RuntimeError(f"Metadta missing for version '{version}'")
     
     with open(metadata_path) as f:
          return json.load(f)

# ----------------------------------------------------
# Main
# ----------------------------------------------------

def main(args):

     logger = logging.getLogger(__name__)
     logger.info("Starting inference pipeline")

     config = load_config(args.config)

     models_path = Path(config["output"]["model_dir"])  
     registry = ModelRegistry(models_path / "registry.json")

     version, source = resolve_model_version(registry, args.version)

     if source == "override":
          logger.warning("Loading OVERRIDE model version '%s'", version)
     
     model = load_model(models_path, version)
     metadata = load_metadata(model_path, version) 

     # Load intference data (placeholder)
     df = pd.read_parquet("data/train.parquet")

     target = metadata["dataset"]["target"]
     X = df.drop(columns=[target])

     validate_schema(X, metadata)

     predictions = model.predict(X)

     log_predictions(X, predictions, version)

     logger.info("Generated %d predictions", len(predictions))
     logger.debug("Sample predictions: %s", predictions[:10])

# ----------------------------------------------------
# Entrypoint
# ----------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    main(args)       
