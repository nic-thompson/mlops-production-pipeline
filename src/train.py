import yaml
import joblib
import logging
import argparse
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def configure_logging(log_level: str):
    log_level = log_level.upper()

    if not hasattr(logging, log_level):
         raise ValueError(f"Invalid log level: {log_level}")
    
    numeric_level = getattr(logging, log_level)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log  level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)

def parse_args():
        parser = argparse.ArgumentParser(description="Training pipeline")

        parser.add_argument(
            "--log-level",
            default="INFO",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        )
        
        parser.add_argument(
            "--config",
            default="config.yaml",
            help="Path to configuration file",
        )

        return parser.parse_args()

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def main(args):
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    logger.info("Starting training pipeline")

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=config["data"]["test_size"], 
        random_state=config["data"]["random_state"]
    )

    logger.debug(
         "Training dataset shape: X_train=%s, X_test=%s", 
            X_train.shape, 
            X_test.shape)

    # Train model
    model = RandomForestClassifier(**config["model"])

    logger.debug("Training parameters: %s", model.get_params())

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logger.info("Model accuracy: %.4f", accuracy)

    # Save model artifact
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / config["output"]["model_name"]

    joblib.dump(model, model_path)

    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    main(args)

