import pandas as pd
from typing import Dict
from scipy.stats import ks_2samp
from datetime import datetime 

class DriftDetector:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
    
    def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Dict[str, float]:
        
        drift_scores = {}
    
        for col in reference_df.columns:
            if col not in current_df.columns:
                continue

            ref = reference_df[col].dropna()
            cur = current_df[col].dropna()

            if ref.empty or cur.empty:
                continue

            stat, p_value = ks_2samp(ref, cur)

            drift_scores[col] = p_value

        return drift_scores
    
    def has_drift(self, drift_scores: Dict[str, float]) -> bool:
        return any(p < self.threshold for p in drift_scores.values())        
    
    # Public API
    def detect(
            self,
            reference_df: pd.DataFrame,
            current_df: pd.DataFrame
    ) -> Dict:

            drift_scores = self.detect_feature_drift(reference_df, current_df)
            drift_detected = self.has_drift(drift_scores)

            return {
                 "timestamp": datetime.utcnow().isoformat(),
                 "threshold": self.threshold,
                 "drift_detected": drift_detected,
                 "feature_p_values": drift_scores,
            }