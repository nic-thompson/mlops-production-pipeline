import argparse
import logging
from pathlib import Path

from registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(
        description=("Rollback the current production model to the previous archived version.")
    )

    parser.add_argument(
        '--registry-path',
        type=str,
        default="artifacts/models/registry.json",
        help="Path to the model registry file",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    registry = ModelRegistry(Path(args.registry_path))

    current_prod = registry.get_production()
    logging.info(f"Current production version: {current_prod}")

    registry.rollback_production()

    new_prod = registry.get_production()
    logging.info(f"Rollback successful. New production version: {new_prod}")

if __name__ == "__main__":
    main()