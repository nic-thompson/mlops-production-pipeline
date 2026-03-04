import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from src.drift import DriftDetector

# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Drift Job
# ---------------------------------------------------------

def run_drift_job(
        reference_path: Path,
        current_path: Path,
        report_path: Path,
        threshold: float = 0.05,
) -> int:
    """
    Runs drift detection and writes a JSON report.

    Returns: 
        0 if no drift detected
        1 if drift detection
        2 if job failure
    """

    try:
        logger.info("Logging datasets...")
        reference = pd.read_parquet(reference_path)
        current = pd.read_parquet(current_path)

        logger.info("Running drift detection...")
        detector = DriftDetector(threshold=threshold)
        scores = detector.detect_feature_drift(reference, current)
        drift_detected = detector.has_drift(scores)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "threshold": threshold,
            "drift_detected": drift_detected,
            "scores": scores,
        }

        logger.info("Writing drift report...")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        if drift_detected:
            logger.warning("Drift detected.")
            return 1

        logger.info("No drift detected.") 
        return 0
    
    except Exception as e:
        logger.exception("Drift job failed.")
        return 2

# ---------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------

if __name__ == "__main__":
    exit_code = run_drift_job(
        reference_path=Path("data/reference.parquet"),
        current_path=Path("data/live.parquet"),
        report_path=Path("reports/drift_report.json"),
        threshold=0.5,
    )

    sys.exit(exit_code)