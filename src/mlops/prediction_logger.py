import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

def log_predictions(X, predictions, model_version):

     now = datetime.now(timezone.utc)

     date_str = now.strftime("%Y-%m-%d")

     log_path = Path("data/predictions") / f"{date_str}.parquet"

     records = X.copy()
     records["prediction"] = predictions
     records["model_version"] = model_version
     records["timestamp"] = datetime.now(timezone.utc).isoformat()

     log_path.parent.mkdir(parents=True, exist_ok=True)

     if log_path.exists():
          existing = pd.read_parquet(log_path)
          records = pd.concat([existing, records], ignore_index=True)
     
     records.to_parquet(log_path)