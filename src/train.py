import logging
import argparse
import pandas as pd
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
        return parser.parse_args()

def main():
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline")

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.debug(
         "Training dataset shape: X_train=%s, X_test=%s", 
            X_train.shape, 
            X_test.shape)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    logger.debug("Training parameters: %s", model.get_params())

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logger.info("Model accuracy: %.4f", accuracy)


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)

    logger = logging.getLogger(__name__)

    main()

