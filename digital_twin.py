"""
Digital Twin Module
Virtual equipment simulation and what-if analysis.
"""

import json
from typing import Dict, List, Any, Optional
from llm_provider import call_llm

SYSTEM_PROMPT = """You are an expert in industrial equipment digital twins and predictive maintenance.
You simulate how equipment behaves under different conditions and predict failure modes.
Provide detailed, technical analysis of equipment behavior under simulated conditions."""


def get_equipment_twin_params(equipment_name: str) -> Dict:
    """Get default simulation parameters for equipment type."""
    name_lower = equipment_name.lower()
    
    equipment_params = {
        "pump": {
            "temperature_range": (40, 120),
            "pressure_range": (50, 200),
            "vibration_range": (0.1, 5.0),
            "rpm_range": (1000, 3600),
            "critical_temp": 85,
            "critical_vibration": 3.0,
            "critical_pressure": 150,
        },
        "motor": {
            "temperature_range": (40, 150),
            "vibration_range": (0.1, 4.0),
            "rpm_range": (500, 3600),
            "current_range": (5, 50),
            "critical_temp": 100,
            "critical_vibration": 2.5,
        },
        "compressor": {
            "temperature_range": (50, 180),
            "pressure_range": (100, 250),
            "vibration_range": (0.1, 4.0),
            "rpm_range": (500, 2000),
            "critical_temp": 120,
            "critical_vibration": 2.5,
            "critical_pressure": 200,
        },
        "fan": {
            "temperature_range": (30, 80),
            "vibration_range": (0.1, 3.0),
            "rpm_range": (500, 1800),
            "airflow_range": (100, 1000),
            "critical_temp": 65,
            "critical_vibration": 2.0,
        },
        "gearbox": {
            "temperature_range": (40, 100),
            "vibration_range": (0.1, 4.0),
            "rpm_range": (100, 1000),
            "critical_temp": 75,
            "critical_vibration": 2.5,
        },
    }
    
    for key, params in equipment_params.items():
        if key in name_lower:
            return {"equipment_type": key, **params}
    
    return {
        "equipment_type": "generic",
        "temperature_range": (40, 100),
        "vibration_range": (0.1, 3.0),
        "rpm_range": (500, 2000),
        "critical_temp": 75,
        "critical_vibration": 2.5,
    }


def analyze_digital_twin(
    equipment_name: str,
    simulated_params: Dict[str, Any],
    current_state: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Analyze equipment behavior under simulated conditions using AI."""
    
    params_json = json.dumps(simulated_params, indent=2)
    current_json = json.dumps(current_state, indent=2) if current_state else "No current state data"
    
    prompt = f"""You are analyzing a digital twin simulation for equipment: {equipment_name}

## Simulated Parameters
{params_json}

## Current Equipment State (for comparison)
{current_json}

Provide a detailed digital twin analysis in JSON format:
1. "behavior_prediction": How the equipment will behave under these conditions
2. "risk_assessment": Risk level and failure probability with explanation
3. "warning_signs": What warning signs would appear
4. "expected_failures": What components are most likely to fail first
5. "maintenance_recommendations": What maintenance should be performed
6. "operational_impact": How operations would be affected
7. "safety_concerns": Any safety issues to be aware of

Consider:
- Temperature exceeding critical limits → thermal stress, seal degradation, lubricant breakdown
- High vibration → bearing wear, misalignment, loose components
- Pressure variations → seal failure, valve issues
- Sustained abnormal conditions → accelerated degradation, eventual failure"""

    response = call_llm(prompt, SYSTEM_PROMPT, max_tokens=2000)
    
    if response:
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            return {"success": True, "analysis": json.loads(json_str), "source": "ai"}
        except (json.JSONDecodeError, IndexError):
            return {"success": True, "analysis": {"raw_response": response}, "source": "ai"}
    
    return _rule_based_analysis(equipment_name, simulated_params)


def _rule_based_analysis(equipment_name: str, params: Dict) -> Dict:
    """Rule-based analysis when AI is not available."""
    
    temp = params.get("temperature", 70)
    vibration = params.get("vibration", 1.0)
    pressure = params.get("pressure", 100)
    
    params = get_equipment_twin_params(equipment_name)
    critical_temp = params.get("critical_temp", 75)
    critical_vibration = params.get("critical_vibration", 2.5)
    
    risks = []
    risk_score = 0
    
    if temp > critical_temp:
        risk_score += 40
        risks.append(f"Temperature {temp}°F exceeds critical {critical_temp}°F - thermal stress on components")
    elif temp > critical_temp * 0.85:
        risk_score += 20
        risks.append(f"Temperature elevated - monitor for degradation")
    
    if vibration > critical_vibration:
        risk_score += 35
        risks.append(f"Vibration {vibration}mm/s exceeds critical {critical_vibration}mm/s - bearing damage likely")
    elif vibration > critical_vibration * 0.8:
        risk_score += 15
        risks.append(f"Increased vibration detected")
    
    if risk_score >= 70:
        risk_level = "CRITICAL"
    elif risk_score >= 40:
        risk_level = "HIGH"
    elif risk_score >= 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "success": True,
        "analysis": {
            "risk_assessment": {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "factors": risks,
            },
            "behavior_prediction": f"At {temp}°F and {vibration}mm/s vibration, equipment shows {'significant stress' if risk_score > 40 else 'normal'} behavior.",
            "warning_signs": [
                "Increased bearing temperature" if vibration > critical_vibration * 0.8 else "None detected",
                "Unusual noise patterns" if vibration > critical_vibration else "None detected",
            ],
            "expected_failures": ["Bearing failure", "Seal degradation", "Lubricant breakdown"] if risk_score > 40 else ["Minor wear"],
            "maintenance_recommendations": [
                "Immediate inspection required" if risk_score > 60 else "Scheduled monitoring",
                "Check bearing condition",
                "Verify lubricant levels",
            ],
            "operational_impact": "Reduced efficiency and potential downtime" if risk_score > 40 else "Normal operations",
            "safety_concerns": "Stop operation if conditions worsen" if risk_score > 60 else "Continue with monitoring",
        },
        "source": "rule_based",
    }


def get_twin_visualization_data(
    equipment_name: str,
    params: Dict,
) -> Dict:
    """Generate visualization data for the digital twin."""
    
    params_config = get_equipment_twin_params(equipment_name)
    
    temp = params.get("temperature", 70)
    vibration = params.get("vibration", 1.0)
    pressure = params.get("pressure", 100)
    rpm = params.get("rpm", 1500)
    
    temp_pct = (temp - params_config["temperature_range"][0]) / (params_config["temperature_range"][1] - params_config["temperature_range"][0])
    vib_pct = (vibration - params_config["vibration_range"][0]) / (params_config["vibration_range"][1] - params_config["vibration_range"][0])
    
    health_score = 100 - (temp_pct * 40) - (vib_pct * 40)
    health_score = max(0, min(100, health_score))
    
    if health_score >= 70:
        status = "healthy"
        color = "#22C55E"
    elif health_score >= 40:
        status = "warning"
        color = "#F59E0B"
    else:
        status = "critical"
        color = "#DC2626"
    
    return {
        "health_score": round(health_score, 1),
        "status": status,
        "color": color,
        "gauge_data": {
            "temperature": {"value": temp, "min": params_config["temperature_range"][0], "max": params_config["temperature_range"][1], "critical": params_config.get("critical_temp", 75)},
            "vibration": {"value": vibration, "min": params_config["vibration_range"][0], "max": params_config["vibration_range"][1], "critical": params_config.get("critical_vibration", 2.5)},
        },
        "simulation_params": params,
        "equipment_type": params_config.get("equipment_type", "generic"),
    }
