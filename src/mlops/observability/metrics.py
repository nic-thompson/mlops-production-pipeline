from prometheus_client import Counter, Gauge

predictions_total = Counter(
    "mlops_predictions_total",
    "Total number of predictions served"
)

drift_score = Gauge(
    "mlops_drift_score",
    "Current drift score detected by monitoring"
)

retraining_runs_total = Counter(
    "mlops_retraining_runs_total",
    "Total number of retraining runs triggered"
)
