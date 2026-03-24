"""
ML Features Module
Provides anomaly detection (Isolation Forest), failure prediction,
and equipment comparison utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


def detect_anomalies_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Isolation Forest to detect anomalies in sensor readings.
    Returns a copy of the dataframe with an 'anomaly' column (-1 = anomaly, 1 = normal).
    """
    from sklearn.ensemble import IsolationForest

    if df.empty:
        return df.copy()

    df_out = df.copy()
    numeric_cols = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in df.columns]

    if not numeric_cols:
        df_out["anomaly"] = 1
        df_out["anomaly_score"] = 0.0
        return df_out

    features = df_out[numeric_cols].copy()
    features = features.fillna(features.median())

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
    )
    df_out["anomaly"] = model.fit_predict(features)
    df_out["anomaly_score"] = model.decision_function(features)

    return df_out


def predict_failure_probability(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Predict failure probability for each equipment using a simple ML model
    based on recent sensor trends and event history.
    """
    if df.empty:
        return []

    predictions = []
    equipment_ids = df["equipment_id"].unique()

    for eq_id in equipment_ids:
        eq_data = df[df["equipment_id"] == eq_id].copy()
        eq_name = eq_data["equipment_name"].iloc[0] if not eq_data.empty else eq_id

        # Feature engineering
        features = {}

        # Recent critical events ratio
        total = len(eq_data)
        critical_count = len(eq_data[eq_data["severity"] == "critical"])
        warning_count = len(eq_data[eq_data["severity"] == "warning"])
        features["critical_ratio"] = critical_count / max(total, 1)
        features["warning_ratio"] = warning_count / max(total, 1)

        # Temperature trend
        if "temperature" in eq_data.columns:
            temps = eq_data["temperature"].dropna().values[-5:]
            if len(temps) >= 2:
                features["temp_trend"] = float(np.polyfit(range(len(temps)), temps, 1)[0])
                features["temp_max"] = float(np.max(temps))
                features["temp_std"] = float(np.std(temps))
            else:
                features["temp_trend"] = 0.0
                features["temp_max"] = float(temps[0]) if len(temps) > 0 else 0.0
                features["temp_std"] = 0.0

        # Vibration trend
        if "vibration" in eq_data.columns:
            vibs = eq_data["vibration"].dropna().values[-5:]
            if len(vibs) >= 2:
                features["vib_trend"] = float(np.polyfit(range(len(vibs)), vibs, 1)[0])
                features["vib_max"] = float(np.max(vibs))
                features["vib_std"] = float(np.std(vibs))
            else:
                features["vib_trend"] = 0.0
                features["vib_max"] = float(vibs[0]) if len(vibs) > 0 else 0.0
                features["vib_std"] = 0.0

        # Calculate composite failure probability
        prob = 5.0  # base probability

        # Critical events weigh heavily
        prob += features["critical_ratio"] * 40
        prob += features["warning_ratio"] * 15

        # Rising temperature is a strong signal
        if features.get("temp_trend", 0) > 0.5:
            prob += min(20, features["temp_trend"] * 10)
        if features.get("temp_max", 0) > 85:
            prob += 15

        # Rising vibration
        if features.get("vib_trend", 0) > 0.1:
            prob += min(15, features["vib_trend"] * 20)
        if features.get("vib_max", 0) > 3.0:
            prob += 10

        # High variability
        if features.get("temp_std", 0) > 10:
            prob += 5
        if features.get("vib_std", 0) > 1.0:
            prob += 5

        prob = max(2, min(95, prob))

        # Risk level
        if prob >= 70:
            risk = "CRITICAL"
            action = "Immediate inspection required"
            eta = "24-48 hours"
        elif prob >= 50:
            risk = "HIGH"
            action = "Schedule preventive maintenance"
            eta = "1-2 weeks"
        elif prob >= 30:
            risk = "MEDIUM"
            action = "Monitor closely"
            eta = "1-3 months"
        else:
            risk = "LOW"
            action = "Continue normal operation"
            eta = "3+ months"

        # Contributing factors
        contributing = []
        if features["critical_ratio"] > 0:
            contributing.append(f"Critical event history ({critical_count} events)")
        if features.get("temp_trend", 0) > 0.5:
            contributing.append(f"Rising temperature trend (+{features['temp_trend']:.1f}/reading)")
        if features.get("vib_trend", 0) > 0.1:
            contributing.append(f"Increasing vibration (+{features['vib_trend']:.2f}/reading)")
        if features.get("temp_max", 0) > 80:
            contributing.append(f"High peak temperature ({features['temp_max']:.1f} F)")
        if features.get("vib_max", 0) > 2.5:
            contributing.append(f"High peak vibration ({features['vib_max']:.1f} mm/s)")
        if not contributing:
            contributing.append("No significant risk factors detected")

        predictions.append(
            {
                "equipment_id": eq_id,
                "equipment_name": eq_name,
                "failure_probability": round(prob, 1),
                "risk_level": risk,
                "recommended_action": action,
                "estimated_time_to_failure": eta,
                "contributing_factors": contributing,
                "features": features,
            }
        )

    predictions.sort(key=lambda x: x["failure_probability"], reverse=True)
    return predictions




def generate_report_data(
    df: pd.DataFrame,
    patterns: List[Dict],
    near_misses: List[Dict],
    schedule: List[Dict],
    health_scores: List[Dict],
    analysis: Dict,
) -> str:
    """Generate a text report suitable for PDF export."""
    lines = []
    lines.append("=" * 60)
    lines.append("PREDICTIVE MAINTENANCE REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # Executive Summary
    if analysis and analysis.get("success"):
        a = analysis.get("analysis", {})
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        if isinstance(a, dict) and "executive_summary" in a:
            lines.append(a["executive_summary"])
        elif isinstance(a, dict) and "raw_response" in a:
            lines.append(a["raw_response"][:500])
        lines.append("")

    # Equipment Health
    lines.append("EQUIPMENT HEALTH SCORES")
    lines.append("-" * 40)
    for h in health_scores:
        lines.append(f"  {h['equipment_name']:30s} Score: {h['health_score']:3d}/100  Status: {h['status'].upper()}")
    lines.append("")

    # Degradation Patterns
    lines.append(f"DEGRADATION PATTERNS ({len(patterns)} detected)")
    lines.append("-" * 40)
    for p in patterns:
        lines.append(f"  [{p['severity'].upper():8s}] {p['equipment_name']}: {p['signal']}")
        lines.append(f"            {p['description']}")
        lines.append(f"            Confidence: {p['confidence']}%")
        lines.append(f"            Action: {p['recommendation']}")
        lines.append("")

    # Near-Miss Events
    lines.append(f"NEAR-MISS EVENTS ({len(near_misses)} detected)")
    lines.append("-" * 40)
    for nm in near_misses:
        lines.append(f"  {nm['equipment_name']}: {nm['parameter']}")
        lines.append(f"    {nm['description']}")
        lines.append(f"    Failure Probability: {nm['failure_probability']}%")
        lines.append(f"    Action: {nm['action_required']}")
        lines.append("")

    # Maintenance Schedule
    lines.append(f"MAINTENANCE SCHEDULE ({len(schedule)} tasks)")
    lines.append("-" * 40)
    for i, s in enumerate(schedule):
        lines.append(f"  {i+1}. [{s['urgency']:10s}] {s['equipment_name']}: {s['task']}")
        lines.append(f"     Confidence: {s['confidence']}% | By: {s['recommended_date']} | Duration: {s['estimated_duration']}")
        lines.append("")

    # Data Summary
    lines.append("DATA SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total Records: {len(df)}")
    lines.append(f"  Equipment Count: {df['equipment_id'].nunique() if 'equipment_id' in df.columns else 'N/A'}")
    if "timestamp" in df.columns:
        lines.append(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")

    return "\n".join(lines)
