import logging
import sys
from pathlib import Path

from src.registry import ModelRegistry

# ---------------------------------
# Logging
# ---------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------
# Paths
# ---------------------------------

REGISTRY_PATH = Path("artifacts/models/registry.json")

# ---------------------------------
# Promotion Job
# ---------------------------------

def promote_model():

    try:

        logger.info("Loading model registry...")
        registry = ModelRegistry(REGISTRY_PATH)

        staging_version = registry.get_staging

        if staging_version is None:
            logger.error("No staging model available to promote.")
            return 1
        
        logger.info(f"Model {staging_version} successfully promoted to production.")

        return 0
    
    except Exception:
        logger.exception("Model promotion failed.")
        return 2
    
# ---------------------------------
# Entry Point
# ---------------------------------

if __name__ == "__main__":
    exit_code = promote_model()
    sys.exit(exit_code)