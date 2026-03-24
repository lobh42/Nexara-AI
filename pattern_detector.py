"""
Pattern Detector Module
Identifies degradation signals, near-miss events, and equipment dependencies.
Generates prioritized maintenance schedules with confidence scores and explanations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict


def detect_degradation_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect degradation signals that precede failures.
    Looks for gradual increases in temperature, vibration, or pressure before critical events.
    """
    patterns = []

    if df.empty:
        return patterns

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    equipment_ids = df["equipment_id"].unique()

    for eq_id in equipment_ids:
        eq_data = df[df["equipment_id"] == eq_id].copy()
        eq_name = eq_data["equipment_name"].iloc[0] if not eq_data.empty else eq_id

        # Check for temperature escalation
        if "temperature" in eq_data.columns:
            temps = eq_data[eq_data["temperature"] > 0]["temperature"].values
            if len(temps) >= 3:
                # Look for consistent upward trend
                diffs = np.diff(temps)
                rising_count = np.sum(diffs > 0)
                if rising_count >= len(diffs) * 0.6 and len(diffs) >= 2:
                    # find max sequential increase
                    max_increase = 0
                    start_val = temps[0]
                    end_val = temps[0]
                    for i in range(len(temps)):
                        for j in range(i+1, min(i+10, len(temps))): # look ahead up to 10 readings
                            if temps[j] - temps[i] > max_increase:
                                max_increase = temps[j] - temps[i]
                                start_val = temps[i]
                                end_val = temps[j]
                                
                    temp_increase = max_increase
                    
                    if temp_increase > 5:
                        confidence = min(95, int(40 + (temp_increase / max(start_val, 1)) * 200 + rising_count * 5))
                        confidence = max(20, min(confidence, 98))

                        # Check if there was a critical event after
                        critical_events = eq_data[eq_data["severity"] == "critical"]
                        is_confirmed = len(critical_events) > 0

                        patterns.append({
                            "equipment_id": eq_id,
                            "equipment_name": eq_name,
                            "pattern_type": "temperature_escalation",
                            "signal": "Temperature Degradation",
                            "description": f"Temperature rose from {start_val:.1f}°F to {end_val:.1f}°F ({temp_increase:+.1f}°F) over recent readings",
                            "confidence": confidence,
                            "severity": "critical" if is_confirmed else "warning",
                            "confirmed_failure": is_confirmed,
                            "readings": {
                                "start": float(start_val),
                                "end": float(end_val),
                                "peak": float(np.max(temps)),
                                "trend": "rising"
                            },
                            "recommendation": f"Schedule immediate inspection of {eq_name}. Temperature trend indicates potential thermal failure." if confidence > 70 else f"Monitor {eq_name} temperature closely. Early signs of thermal stress detected.",
                            "evidence": [
                                f"Temperature increased by {temp_increase:.1f}°F across consecutive measurements",
                                f"{rising_count} out of {len(diffs)} consecutive readings showed temperature increase",
                                f"Peak temperature reached {np.max(temps):.1f}°F",
                                "Pattern matches known pre-failure thermal signature" if is_confirmed else "Pattern consistent with early thermal stress"
                            ]
                        })

        # Check for vibration escalation
        if "vibration" in eq_data.columns:
            vibs = eq_data[eq_data["vibration"] > 0]["vibration"].values
            if len(vibs) >= 3:
                diffs = np.diff(vibs)
                rising_count = np.sum(diffs > 0)
                if rising_count >= len(diffs) * 0.5 and len(diffs) >= 2:
                    # find max sequential increase
                    max_increase = 0
                    start_val = vibs[0]
                    end_val = vibs[0]
                    for i in range(len(vibs)):
                        for j in range(i+1, min(i+10, len(vibs))):
                            if vibs[j] - vibs[i] > max_increase:
                                max_increase = vibs[j] - vibs[i]
                                start_val = vibs[i]
                                end_val = vibs[j]
                                
                    vib_increase = max_increase
                    
                    if vib_increase > 0.5:
                        confidence = min(95, int(35 + (vib_increase / max(start_val, 0.1)) * 150 + rising_count * 8))
                        confidence = max(20, min(confidence, 98))

                        critical_events = eq_data[eq_data["severity"] == "critical"]
                        is_confirmed = len(critical_events) > 0

                        patterns.append({
                            "equipment_id": eq_id,
                            "equipment_name": eq_name,
                            "pattern_type": "vibration_escalation",
                            "signal": "Vibration Anomaly",
                            "description": f"Vibration increased from {start_val:.1f} to {end_val:.1f} mm/s ({vib_increase:+.1f}) over recent readings",
                            "confidence": confidence,
                            "severity": "critical" if is_confirmed else "warning",
                            "confirmed_failure": is_confirmed,
                            "readings": {
                                "start": float(start_val),
                                "end": float(end_val),
                                "peak": float(np.max(vibs)),
                                "trend": "rising"
                            },
                            "recommendation": f"Immediate bearing/mechanical inspection for {eq_name}. Vibration pattern indicates imminent mechanical failure." if confidence > 70 else f"Schedule vibration analysis for {eq_name} within 48 hours.",
                            "evidence": [
                                f"Vibration increased by {vib_increase:.1f} mm/s across consecutive measurements",
                                f"{rising_count} out of {len(diffs)} consecutive readings showed vibration increase",
                                f"Peak vibration reached {np.max(vibs):.1f} mm/s",
                                "Exceeds ISO 10816 vibration severity threshold" if np.max(vibs) > 3.0 else "Approaching vibration warning threshold"
                            ]
                        })

        # Check for pressure anomalies
        if "pressure" in eq_data.columns:
            pressures = eq_data[eq_data["pressure"] > 0]["pressure"].values
            if len(pressures) >= 3:
                pressure_std = np.std(pressures)
                pressure_mean = np.mean(pressures)
                if pressure_std > pressure_mean * 0.05 and pressure_mean > 0:
                    confidence = min(85, int(30 + (pressure_std / pressure_mean) * 500))
                    confidence = max(20, min(confidence, 90))

                    patterns.append({
                        "equipment_id": eq_id,
                        "equipment_name": eq_name,
                        "pattern_type": "pressure_instability",
                        "signal": "Pressure Fluctuation",
                        "description": f"Pressure readings showing instability (mean: {pressure_mean:.1f} PSI, std: {pressure_std:.1f} PSI)",
                        "confidence": confidence,
                        "severity": "warning",
                        "confirmed_failure": False,
                        "readings": {
                            "mean": float(pressure_mean),
                            "std": float(pressure_std),
                            "min": float(np.min(pressures)),
                            "max": float(np.max(pressures)),
                            "trend": "unstable"
                        },
                        "recommendation": f"Check pressure system components for {eq_name}. Investigate seals, valves, and fluid levels.",
                        "evidence": [
                            f"Pressure standard deviation of {pressure_std:.1f} PSI exceeds normal range",
                            f"Pressure varied between {np.min(pressures):.1f} and {np.max(pressures):.1f} PSI",
                            f"Coefficient of variation: {(pressure_std/pressure_mean*100):.1f}%"
                        ]
                    })

    # Sort by confidence (highest first)
    patterns.sort(key=lambda x: x["confidence"], reverse=True)
    return patterns


def detect_near_misses(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identify near-miss events: situations where parameters approached dangerous levels
    but didn't result in failure (yet).
    """
    near_misses = []

    if df.empty:
        return near_misses

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    thresholds = {
        "temperature": {"warning": 75, "critical": 85, "unit": "°F"},
        "vibration": {"warning": 2.0, "critical": 3.5, "unit": "mm/s"},
        "pressure_high": {"warning": 175, "critical": 190, "unit": "PSI"},
    }

    equipment_ids = df["equipment_id"].unique()

    for eq_id in equipment_ids:
        eq_data = df[df["equipment_id"] == eq_id]
        eq_name = eq_data["equipment_name"].iloc[0] if not eq_data.empty else eq_id

        # Check each threshold parameter
        if "temperature" in eq_data.columns:
            high_temp = eq_data[
                (eq_data["temperature"] >= thresholds["temperature"]["warning"]) &
                (eq_data["temperature"] < thresholds["temperature"]["critical"]) &
                (eq_data["severity"] != "critical")
            ]
            if len(high_temp) > 0:
                max_temp = high_temp["temperature"].max()
                closeness = (max_temp - thresholds["temperature"]["warning"]) / (thresholds["temperature"]["critical"] - thresholds["temperature"]["warning"])
                probability = min(85, int(30 + closeness * 55))

                near_misses.append({
                    "equipment_id": eq_id,
                    "equipment_name": eq_name,
                    "parameter": "Temperature",
                    "current_value": float(max_temp),
                    "warning_threshold": thresholds["temperature"]["warning"],
                    "critical_threshold": thresholds["temperature"]["critical"],
                    "unit": thresholds["temperature"]["unit"],
                    "failure_probability": probability,
                    "occurrences": len(high_temp),
                    "last_occurrence": str(high_temp["timestamp"].max()),
                    "status": "approaching_critical" if closeness > 0.7 else "elevated",
                    "description": f"Temperature reached {max_temp:.1f}°F ({closeness*100:.0f}% toward critical threshold)",
                    "action_required": "Immediate inspection recommended" if closeness > 0.7 else "Schedule inspection within 72 hours"
                })

        if "vibration" in eq_data.columns:
            high_vib = eq_data[
                (eq_data["vibration"] >= thresholds["vibration"]["warning"]) &
                (eq_data["vibration"] < thresholds["vibration"]["critical"]) &
                (eq_data["severity"] != "critical")
            ]
            if len(high_vib) > 0:
                max_vib = high_vib["vibration"].max()
                closeness = (max_vib - thresholds["vibration"]["warning"]) / (thresholds["vibration"]["critical"] - thresholds["vibration"]["warning"])
                probability = min(80, int(25 + closeness * 55))

                near_misses.append({
                    "equipment_id": eq_id,
                    "equipment_name": eq_name,
                    "parameter": "Vibration",
                    "current_value": float(max_vib),
                    "warning_threshold": thresholds["vibration"]["warning"],
                    "critical_threshold": thresholds["vibration"]["critical"],
                    "unit": thresholds["vibration"]["unit"],
                    "failure_probability": probability,
                    "occurrences": len(high_vib),
                    "last_occurrence": str(high_vib["timestamp"].max()),
                    "status": "approaching_critical" if closeness > 0.7 else "elevated",
                    "description": f"Vibration reached {max_vib:.1f} mm/s ({closeness*100:.0f}% toward critical threshold)",
                    "action_required": "Immediate mechanical inspection" if closeness > 0.7 else "Schedule vibration analysis"
                })

    near_misses.sort(key=lambda x: x["failure_probability"], reverse=True)
    return near_misses




def generate_maintenance_schedule(
    df: pd.DataFrame,
    patterns: List[Dict],
    near_misses: List[Dict],
    dependencies: Dict
) -> List[Dict[str, Any]]:
    """
    Generate a prioritized maintenance schedule with explanations.
    Each task has a confidence score and reasoning.
    """
    schedule = []
    now = datetime.now()

    # Priority 1: Critical patterns (confirmed failures or high confidence degradation)
    for pattern in patterns:
        if pattern["confidence"] >= 70 or pattern["severity"] == "critical":
            urgency = "IMMEDIATE" if pattern["confidence"] >= 80 else "HIGH"
            schedule.append({
                "priority": 1,
                "urgency": urgency,
                "equipment_id": pattern["equipment_id"],
                "equipment_name": pattern["equipment_name"],
                "task": f"Inspect and repair: {pattern['signal']}",
                "description": pattern["description"],
                "confidence": pattern["confidence"],
                "recommended_date": (now + timedelta(hours=4 if urgency == "IMMEDIATE" else 24)).strftime("%Y-%m-%d %H:%M"),
                "estimated_duration": "2-4 hours",
                "reasoning": pattern["recommendation"],
                "evidence": pattern["evidence"],
                "affected_equipment": [
                    eq for eq in dependencies.get("impact_matrix", {}).get(pattern["equipment_id"], {}).get("affected_equipment", [])
                ],
                "production_impact": "High - Line stoppage likely if not addressed" if urgency == "IMMEDIATE" else "Medium - Performance degradation expected"
            })

    # Priority 2: Near-miss events
    for nm in near_misses:
        if nm["failure_probability"] >= 40:
            schedule.append({
                "priority": 2,
                "urgency": "MEDIUM",
                "equipment_id": nm["equipment_id"],
                "equipment_name": nm["equipment_name"],
                "task": f"Preventive check: {nm['parameter']} anomaly",
                "description": nm["description"],
                "confidence": nm["failure_probability"],
                "recommended_date": (now + timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
                "estimated_duration": "1-2 hours",
                "reasoning": nm["action_required"],
                "evidence": [
                    f"{nm['parameter']} at {nm['current_value']}{nm['unit']} (warning: {nm['warning_threshold']}, critical: {nm['critical_threshold']})",
                    f"Occurred {nm['occurrences']} time(s), last: {nm['last_occurrence']}"
                ],
                "affected_equipment": [],
                "production_impact": "Low-Medium - Equipment can run but at risk"
            })

    # Priority 3: Moderate patterns
    for pattern in patterns:
        if pattern["confidence"] < 70 and pattern["severity"] != "critical":
            existing = [s for s in schedule if s["equipment_id"] == pattern["equipment_id"]]
            if not existing:
                schedule.append({
                    "priority": 3,
                    "urgency": "LOW",
                    "equipment_id": pattern["equipment_id"],
                    "equipment_name": pattern["equipment_name"],
                    "task": f"Monitor and evaluate: {pattern['signal']}",
                    "description": pattern["description"],
                    "confidence": pattern["confidence"],
                    "recommended_date": (now + timedelta(days=7)).strftime("%Y-%m-%d %H:%M"),
                    "estimated_duration": "30 minutes - 1 hour",
                    "reasoning": pattern["recommendation"],
                    "evidence": pattern["evidence"],
                    "affected_equipment": [],
                    "production_impact": "Low - Can be scheduled during planned downtime"
                })

    schedule.sort(key=lambda x: (x["priority"], -x["confidence"]))
    return schedule


def get_equipment_health_scores(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calculate overall health scores for each piece of equipment."""
    if df.empty:
        return []

    health_scores = []
    equipment_ids = df["equipment_id"].unique()

    for eq_id in equipment_ids:
        eq_data = df[df["equipment_id"] == eq_id]
        eq_name = eq_data["equipment_name"].iloc[0] if not eq_data.empty else eq_id

        # Start with 100 and deduct points
        score = 100

        # Deduct for critical events
        critical_count = len(eq_data[eq_data["severity"] == "critical"])
        score -= critical_count * 15

        # Deduct for warnings
        warning_count = len(eq_data[eq_data["severity"] == "warning"])
        score -= warning_count * 5

        # Deduct for maintenance events
        maint_count = len(eq_data[eq_data["log_type"] == "maintenance"])
        score -= maint_count * 3

        # Check recent trend (last entries)
        recent = eq_data.tail(3)
        if len(recent) > 0:
            recent_critical = len(recent[recent["severity"] == "critical"])
            if recent_critical > 0:
                score -= 10

        score = max(0, min(100, score))

        # Determine status
        if score >= 80:
            status = "healthy"
        elif score >= 60:
            status = "attention"
        elif score >= 40:
            status = "warning"
        else:
            status = "critical"

        health_scores.append({
            "equipment_id": eq_id,
            "equipment_name": eq_name,
            "health_score": score,
            "status": status,
            "total_logs": len(eq_data),
            "critical_events": int(critical_count),
            "warning_events": int(warning_count),
            "maintenance_events": int(maint_count),
            "last_log": str(eq_data["timestamp"].max()) if not eq_data.empty else None
        })

    health_scores.sort(key=lambda x: x["health_score"])
    return health_scores


def get_trend_data(df: pd.DataFrame, equipment_id: Optional[str] = None) -> Dict[str, Any]:
    """Get time-series trend data for charts."""
    if df.empty:
        return {"timestamps": [], "series": {}}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

    if equipment_id:
        df = df[df["equipment_id"] == equipment_id]

    timestamps = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    series = {}
    for col in ["temperature", "vibration", "pressure", "rpm"]:
        if col in df.columns:
            series[col] = df[col].tolist()

    severity_counts = df.groupby(df["timestamp"].dt.date)["severity"].value_counts().unstack(fill_value=0)
    severity_timeline = {}
    if not severity_counts.empty:
        for sev in ["info", "warning", "critical"]:
            if sev in severity_counts.columns:
                severity_timeline[sev] = severity_counts[sev].tolist()

    return {
        "timestamps": timestamps,
        "dates": [str(d) for d in sorted(df["timestamp"].dt.date.unique())],
        "series": series,
        "severity_timeline": severity_timeline,
        "equipment_ids": df["equipment_id"].unique().tolist() if not equipment_id else [equipment_id]
    }
