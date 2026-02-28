import argparse
import logging
from pathlib import Path
import yaml

from registry import ModelRegistry

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Promote model version")

    parser.add_argument(
        "--version",
        required=True,
        help="Model version to promote",
    )

    parser.add_argument(
        "--to",
        required=True,
        choices=["staging", "production"],
        help="Target environement",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file.",
    )

    return parser.parse_args()

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    configure_logging()
    args = parse_args()

    config = load_config(args.config)
    models_base_path = Path(config["output"]["model_dir"])
    registry_path = models_base_path / "registry.json"

    registry = ModelRegistry(registry_path)

    if args.to == "staging":
        registry.promote_to_staging(args.version)
        logging.info("Promoted version '%s' to STAGING", args.version)

    elif args.to == "production":
        registry.promote_to_production(args.version)
        logging.info("Promoted version '%s' to PRODUCTION", args.version)

if __name__ == "__main__":
    main()