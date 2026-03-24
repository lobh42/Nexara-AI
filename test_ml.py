import sys
import pandas as pd
from ml_features import detect_anomalies_isolation_forest, predict_failure_probability

df = pd.DataFrame({
    "equipment_id": ["E1", "E2"],
    "equipment_name": ["Pump A", "Pump B"],
    "temperature": [70, 80],
    "vibration": [1.5, 2.5],
    "pressure": [100, 110],
    "rpm": [1500, 1550],
    "severity": ["info", "warning"]
})

print("Testing detect_anomalies_isolation_forest...")
result1 = detect_anomalies_isolation_forest(df)

print("Testing predict_failure_probability...")
result2 = predict_failure_probability(df)

print("SUCCESS")
