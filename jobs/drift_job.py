import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from src.drift import DriftDetector

def parse_args():
    parser = argparse.ArgumentParser(description="Run drift detection job")

    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference dataset (parquet)"
    )

    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current dataset (parquet)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Drift detection threshold (default=0.2)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
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
        current_path: str,
        threshold: float,
        output_path: str
) -> int:
    try:
        logger.info("Logging datasets...")

        reference_df = pd.read_parquet(reference_path)
        current_df = pd.read_parquet(current_path)

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
            logger.info("Drift detected.")
            return 1
        else:
            logger.info("No drift detected")
            return 0
    
    except Exception:
        logger.exception("Drift job failed.")
        return 2
    
# ---------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------

def main():
    args = parse_args()
    
    exit_code = run_drift_job(
        reference_path=args.reference,
        current_path=args.current,
        threshold=args.threshold,
        output_path=args.output
    )

    sys.exit(exit_code)
    
if __name__ == "__main__":
    main()
    