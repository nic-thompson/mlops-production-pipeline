import logging
import sys
from pathlib import Path

from src.mlops.registry import ModelRegistry

# -------------------------------
# Logging
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# -------------------------------
# Paths
# -------------------------------

REGISTRY_PATH = Path("artifacts/models/registry.json")

# -------------------------------
# Rollback Job
# -------------------------------

def rollback_model():

    try:
        
        logger.info("Loading model registry...")
        registry = ModelRegistry(REGISTRY_PATH)

        current_prod = registry.get_production()

        if current_prod is None:
            logger.error("No production model is currnently deplpyed.")
            return 1
        
        logger.info(f"Current current production model: {current_prod}")
        logger.info("Rolling back to previoue archived model...")

        registry.rollback_production()

        new_prod = registry.get_production()

        logger.info(f"Rollback successful. New production model: {new_prod}")

        return 0
    
    except Exception:
        logger.exception("Rollback failed.")
        return 2

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    exit_code = rollback_model()
    sys.exit(exit_code)



