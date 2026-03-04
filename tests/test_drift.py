import pandas as pd
from drift import DriftDetector

def test_no_drift():
    import pandas as pd
    from drift import DriftDetector

    ref = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    cur = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

    detector = DriftDetector()
    scores = detector.detect_feature_drift(ref, cur)

def test_detects_drift():
    ref = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    cur = pd.DataFrame({"x": [100, 200, 300, 400, 500]}) # <---- driftted

    detector = DriftDetector()
    scores = detector.detect_feature_drift(ref, cur)

    assert detector.has_drift(scores) is True