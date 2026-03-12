import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

ALERT_STATE_FILE = Path("alerts/drift_last_alert.json")
ALERT_COOLDOWN = timedelta(hours=1)

def _alert_allowed():

    if not ALERT_STATE_FILE.exists():
        return True
    
    try:
        with open(ALERT_STATE_FILE) as f:
            data = json.load(f)

        last_alert = datetime.fromisoformat(data["timestamp"])

    except Exception:
        # corrupted or empty state file
        logger.warning("Alert state file corrupted. Resetting.")
        return True

    if datetime.now(timezone.utc) - last_alert > ALERT_COOLDOWN:
        return True
    
    return False

def _record_alert():

    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(ALERT_STATE_FILE, "w") as f:
        json.dump(
            {"timestamp": datetime.now(timezone.utc).isoformat()},
            f
        )

def send_alert(title: str, message: str, severity: str = "warning"):
    """
    Send monitoring alert.
    
    Currently logs the alert but can be extended to integrate with:
    - Slack
    - PagerDuty
    - Email
    - OpsGenie
    """

    if not _alert_allowed():
        logger.info("Alert suppressed due to cooldown.")
        return

    alert = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "title": title,
        "message": message,
    }

    logger.warning("ALERT: %s", alert)

    _record_alert()


