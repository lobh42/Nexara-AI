import sys
from ai_engine import analyze_logs_with_llm

logs_summary = {"total_records": 100, "equipment_count": 2, "date_range": {"start": "2026-01-01", "end": "2026-01-02"}, "severity_distribution": {"info": 90, "warning": 10}}
patterns = []
near_misses = []

try:
    result = analyze_logs_with_llm(logs_summary, patterns, near_misses)
    print("SUCCESS")
    print(result)
except Exception as e:
    import traceback
    print("ERROR")
    traceback.print_exc()

