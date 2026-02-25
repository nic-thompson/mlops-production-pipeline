import yaml
import logging
import argparse
import joblib
import pandas as pd
from pathlib import Path
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

    return parser.parse_args()

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(args):
        logger = logging.getLogger(__name__)
        logger.info("Starting inference pipeline")

        config = load_config(args.config)

        # Load model artifact
        model_path = Path(config["output"]["model_dir"]) / config["output"]["model_name"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)

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
