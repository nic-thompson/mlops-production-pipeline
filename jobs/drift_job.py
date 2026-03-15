import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from src.mlops.drift import DriftDetector
from src.mlops.alerts import send_alert
from src.mlops.retraining import trigger_retraining

def parse_args():
    parser = argparse.ArgumentParser(description="Run drift detection job")

    parser.add_argument(
        "--reference",
        default="data/train.parquet",
        help="Reference dataset path"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Drift detection threshold (default=0.2)"
    )

    parser.add_argument(
        "--output",
        default="reports/drift_report.json",
        help="Path to output drift report JSON"
    )

    return parser.parse_args()

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
        reference_path: str,
        threshold: float,
        output_path: str
) -> int:
    try:
        logger.info("Logging datasets...")

        reference_df = pd.read_parquet(reference_path)
        current_df = load_recent_predictions(hours=24)

        # Remove target column if present
        reference_df = reference_df.drop(columns=["target"], errors="ignore")

        # Align columns
        current_df = current_df[reference_df.columns]

        logger.info("Running drift detection...")
        detector = DriftDetector(threshold=threshold)
        report = detector.detect(reference_df, current_df)

        logger.info("Writing drift report...")

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        if report["drift_detected"]:
            send_alert(
                title="Data Drift Detected",
                message=f"Drift detected in features: {report.get('drifted_features', [])}",
                severity="critical"
            )
            logger.info("Drift detected.")

            trigger_retraining()

            return 1
        else:
            logger.info("No drift detected")
            return 0
    
    except Exception:
        logger.exception("Drift job failed.")
        return 2
    
def load_recent_predictions(hours=24):

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    files = list(Path("data/predictions").glob("*.parquet"))

    recent_files = [
        f for f in files
        if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) >= cutoff
    ]

    if not recent_files:
        raise RuntimeError("No recent prediction logs found.")

    dfs = [pd.read_parquet(f) for f in recent_files]

    df = pd.concat(dfs, ignore_index=True)

    # Remove monitoring columns
    df = df.drop(
        columns=["prediction", "model_version", "timestamp", "index"], 
        errors="ignore"
    )  

    return df
  
# ---------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------

def main():
    args = parse_args()
    
    exit_code = run_drift_job(
        reference_path=args.reference,
        threshold=args.threshold,
        output_path=args.output
    )

    sys.exit(exit_code)
    
if __name__ == "__main__":
    main()
    