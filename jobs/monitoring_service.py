import sys
import signal
import time
import logging

from jobs.drift_job import run_drift_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 600 # 10 minutes

running = True

def stop_service(signum, frame):
    global running
    running = False
    logger.info("Stopping monitoring service...")

signal.signal(signal.SIGINT, stop_service)
signal.signal(signal.SIGTERM, stop_service)

def monitoring_loop():

    logger.info("Starting monitoring service")

    while running:

        try:

            run_drift_job(
                reference_path="data/train.parquet",
                threshold=0.5,
                output_path="reports/drift_report.json",
            )
        
        except Exception:
            logger.exception("Monitoring iteration failed")

        logger.info("Sleeping for %s seconds", CHECK_INTERVAL_SECONDS)

        sleep_remaining = CHECK_INTERVAL_SECONDS

        while running and sleep_remaining > 0:
            time.sleep(1)
            sleep_remaining -= 1
        

if __name__ == "__main__":
    monitoring_loop()