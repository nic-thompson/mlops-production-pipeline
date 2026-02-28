import yaml
import logging
import argparse
import joblib
import pandas as pd
from pathlib import Path
from registry import ModelRegistry
from sklearn.datasets import load_breast_cancer


def configure_logging(log_level: str):
    log_level = log_level.upper()

    if not hasattr(logging, log_level):
        raise ValueError(f"Invalid log level: {log_level}")
    
    numeric_level = getattr(logging, log_level)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Inference pipeline")

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file)",
    )

    parser.add_argument(
        "--version",
        type= str,
        default=None,
        help="Explicit model version override (bypasses production)"
    )

    return parser.parse_args()

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_model_version(registry: ModelRegistry, override_version: str | None):
    """
    Decide which model version to load. 
    Returns (version, source) where source is either override or production 
    """

    if override_version:
         if not registry.version_exists(override_version):
              raise RuntimeError(
                   f"Requested version '{override_version}' not found in registry."
              )
         return override_version, "override"
    
    production_version = registry.get_production()
    if production_version is None:
         raise RuntimeError("No production model set in registry.")
    
    return production_version, "production"

def main(args):
        logger = logging.getLogger(__name__)
        logger.info("Starting inference pipeline")

        config = load_config(args.config)
        override_version = args.version #None or str

        # Load model artifact
        models_base_path = Path(config["output"]["model_dir"])  

        registry_path = models_base_path / "registry.json"
        registry = ModelRegistry(registry_path)

        version, source = resolve_model_version(registry, override_version)

        model_path = (
             models_base_path
             / version
             / "model.joblib"
        )

        if not model_path.exists():
             raise FileNotFoundError(
                  f"Model artifact not found for version '{version}': {model_path}"
             )
        
        model = joblib.load(model_path)

        production_version = registry.get_production()

        if source == "override":
          if production_version:
               logger.warning(
                    "Loading OVERRIDE model version '%s' (production is '%s')",
                    version,
                    production_version,
               )
          else:
               logger.warning(
                    "Loading OVERRIDE model version '%s' (no production model set",
                    version,
               )

        # Load dataset (placeholder for real input)
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)

        # Run predictions
        predictions = model.predict(X)

        logger.info("Generated %d predictions", len(predictions))
        logger.debug("Sample predictions: %s", predictions[:10])

if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    main(args)       
