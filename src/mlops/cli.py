import argparse
import subprocess
import sys

def run(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


def main():

    parser = argparse.ArgumentParser(prog="mlops")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Run training pipeline")
    sub.add_parser("predict", help="Run inference pipeline")
    sub.add_parser("drift", help="Run drift detection job")
    sub.add_parser("monitor", help="Start monitoring service")
    sub.add_parser("promote", help="Promote staging model to production")
    sub.add_parser("rollback", help="Rollback production model")

    args = parser.parse_args()

    if args.command == "train":
        run(["python", "-m", "training.pipeline"])

    elif args.command == "predict":
        run(["python", "src/predict.py"])

    elif args.command == "drift":
        run(["python", "-m", "jobs.drift_job"])

    elif args.command == "monitor":
        run(["python", "-m", "jobs.monitoring_service"])

    elif args.command == "promote":
        run(["python", "-m", "jobs.promote_model"])

    elif args.command == "rollback":
        run(["python", "-m", "jobs.rollback_model"])

if __name__ == "__main__":
    main()