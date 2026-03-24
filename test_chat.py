import sys

from ai_engine import chat_query

context = {
    "equipment_health": [{"equipment_id": "EQ1", "equipment_name": "Pump", "status": "healthy", "health_score": 90, "total_logs": 10, "critical_events": 0, "warning_events": 1, "maintenance_events": 0}],
    "patterns": [],
    "schedule": [],
    "near_misses": [],
}

try:
    result = chat_query("How is the pump doing?", context)
    print("SUCCESS")
    print(result)
except Exception as e:
    import traceback
    print("ERROR")
    traceback.print_exc()

