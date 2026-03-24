import sys
import pandas as pd
from pattern_detector import detect_degradation_patterns, detect_near_misses, generate_maintenance_schedule, get_equipment_health_scores
from ml_features import detect_anomalies_isolation_forest, predict_failure_probability
from ai_engine import analyze_logs_with_llm

print("Loading data...")
df = pd.read_csv("dataset/sample_logs.csv")
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "equipment_id"])

print("Running detect_degradation_patterns...")
patterns = detect_degradation_patterns(df)

print("Running detect_near_misses...")
near_misses = detect_near_misses(df)

print("Running get_equipment_health_scores...")
health_scores = get_equipment_health_scores(df)

print("Running generate_maintenance_schedule...")
schedule = generate_maintenance_schedule(df, patterns, near_misses, {})

print("Running detect_anomalies_isolation_forest...")
anomaly_data = detect_anomalies_isolation_forest(df)

print("Running predict_failure_probability...")
failure_predictions = predict_failure_probability(df)

logs_summary = {
    "total_records": len(df),
    "date_range": {
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    },
    "equipment_count": df["equipment_id"].nunique(),
    "severity_distribution": df["severity"].value_counts().to_dict(),
}

print("Running analyze_logs_with_llm...")
analysis = analyze_logs_with_llm(logs_summary, patterns, near_misses)

print("SUCCESS")
