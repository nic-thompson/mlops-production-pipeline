import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

RETRAIN_STATE_FILE = Path("retraining/retrain_state.json")
RETRAIN_COOLDOWN = timedelta(hours=6)

def retrain_allowed():

    if not RETRAIN_STATE_FILE.exists():
        return True
    
    try:
        with open(RETRAIN_STATE_FILE) as f:
            data = json.load(f)

            last_retrain = datetime.fromisoformat(data["timestamp"])

    except Exception:
        logger.warning("Retrain state corrupted. Allowing retrain.")
        return True
    
    return datetime.now(timezone.utc) - last_retrain > RETRAIN_COOLDOWN

def record_retrain():

    RETRAIN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RETRAIN_STATE_FILE, "w") as f:
        json.dump(
            {"timestamp": datetime.now(timezone.utc).isoformat()},
            f
        )

def trigger_retraining():

    if not retrain_allowed():
        logger.info("Retraining surpressed due to cooldown.")
        return
    
    logger.warning("Triggering automatic retraining pipeline.")

    try:
        subprocess.run(
            ["python", "-m", "training.pipeline"],
            check=True
        )
    
    except subprocess.CalledProcessError:
        logger.exception("Retraining pipeline failed.")