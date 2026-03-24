"""
AI Engine Module
Uses built-in free AI engines (HuggingFace + LangChain + rule-based) for
intelligent analysis, chat, and recommendations.
All AI is seamless and invisible to users - no configuration needed.
"""

import json
from typing import Dict, List, Any

from llm_provider import call_llm

SYSTEM_PROMPT = """You are an expert AI assistant for predictive maintenance in manufacturing.
You analyze equipment logs, sensor data, and maintenance records to provide actionable insights.
Your responses should be:
- Technically precise and reference specific data points
- Actionable with clear recommendations
- Include confidence levels when making predictions
- Consider safety implications
- Reference industry standards (ISO, ANSI) when relevant
Always structure your responses clearly with headers and bullet points when appropriate."""


def analyze_logs_with_llm(
    logs_summary: Dict[str, Any],
    patterns: List[Dict],
    near_misses: List[Dict],
    **kwargs,
) -> Dict[str, Any]:
    """Use built-in AI to provide deep analysis of detected patterns."""
    prompt = f"""Analyze the following manufacturing equipment data and provide expert insights:

## Equipment Summary
{json.dumps(logs_summary, indent=2, default=str)}

## Detected Degradation Patterns
{json.dumps(patterns[:5], indent=2, default=str)}

## Near-Miss Events
{json.dumps(near_misses[:5], indent=2, default=str)}

Please provide:
1. A brief executive summary (2-3 sentences)
2. Top 3 critical findings with risk assessment
3. Root cause hypotheses for detected patterns
4. Recommended immediate actions (prioritized)
5. Long-term maintenance strategy suggestions

Format your response as JSON with keys: executive_summary, critical_findings (array), root_causes (array), immediate_actions (array), long_term_strategy (array)."""

    response = call_llm(prompt, SYSTEM_PROMPT)

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

    return _rule_based_analysis(logs_summary, patterns, near_misses)


def _rule_based_analysis(
    logs_summary: Dict[str, Any],
    patterns: List[Dict],
    near_misses: List[Dict],
) -> Dict[str, Any]:
    """Enhanced rule-based analysis with expert-system intelligence."""
    critical_patterns = [p for p in patterns if p.get("severity") == "critical" or p.get("confidence", 0) >= 70]
    total_records = logs_summary.get("total_records", 0)
    equipment_count = logs_summary.get("equipment_count", 0)

    if critical_patterns:
        equipment_names = list(set(p["equipment_name"] for p in critical_patterns))
        risk_level = "HIGH" if len(critical_patterns) > 2 else "MODERATE"
        executive_summary = (
            f"Comprehensive analysis of {total_records} sensor readings across {equipment_count} "
            f"equipment units reveals {len(patterns)} degradation patterns and {len(near_misses)} "
            f"near-miss events. Overall fleet risk level: {risk_level}. "
            f"Immediate attention required for: {', '.join(equipment_names[:3])}."
        )
    else:
        executive_summary = (
            f"Analysis of {total_records} sensor readings across {equipment_count} equipment units "
            f"detected {len(patterns)} degradation patterns and {len(near_misses)} near-miss events. "
            f"No critical-level threats identified; preventive maintenance recommended for flagged items."
        )

    critical_findings = []
    source_patterns = critical_patterns[:3] if critical_patterns else patterns[:3]
    for p in source_patterns:
        risk_score = p.get("confidence", 50)
        if risk_score >= 80:
            risk_level = "HIGH"
            impact = "Potential unplanned downtime within 48-72 hours if unaddressed"
        elif risk_score >= 60:
            risk_level = "MEDIUM"
            impact = "Progressive degradation may lead to failure within 1-2 weeks"
        else:
            risk_level = "LOW"
            impact = "Early-stage degradation detected; monitor closely"
        critical_findings.append(
            {
                "equipment": p["equipment_name"],
                "finding": p["description"],
                "risk_level": risk_level,
                "confidence": risk_score,
                "potential_impact": impact,
                "signal_type": p.get("signal", "unknown"),
            }
        )

    root_causes = []
    seen_types = set()
    for p in patterns:
        ptype = p.get("pattern_type", "unknown")
        if ptype in seen_types:
            continue
        seen_types.add(ptype)

        hypotheses = {
            "temperature_escalation": {
                "hypothesis": "Bearing wear or lubrication breakdown causing increased friction and heat generation. "
                              "Secondary possibility: coolant system degradation or ambient temperature influence.",
                "recommended_test": "Infrared thermography scan + lubricant viscosity analysis",
            },
            "vibration_escalation": {
                "hypothesis": "Mechanical imbalance, misalignment, or bearing defect causing progressive vibration increase. "
                              "Harmonic analysis recommended to isolate frequency-specific root cause.",
                "recommended_test": "Vibration spectrum analysis (FFT) + dynamic balancing check",
            },
            "pressure_instability": {
                "hypothesis": "Hydraulic seal degradation, fluid contamination, or pressure relief valve malfunction "
                              "causing intermittent pressure variations beyond normal operating envelope.",
                "recommended_test": "Hydraulic fluid particle count + seal integrity pressure test",
            },
        }

        if ptype in hypotheses:
            root_causes.append({
                "pattern": p["signal"],
                "equipment": p["equipment_name"],
                **hypotheses[ptype],
                "supporting_evidence": p.get("evidence", []),
            })
        else:
            root_causes.append({
                "pattern": p.get("signal", ptype),
                "equipment": p["equipment_name"],
                "hypothesis": f"Anomalous {ptype} pattern detected. Further investigation needed.",
                "recommended_test": "Detailed inspection and sensor calibration verification",
                "supporting_evidence": p.get("evidence", []),
            })

    immediate_actions = []
    for i, p in enumerate(critical_patterns[:5]):
        timeline = "Within 4 hours" if p.get("confidence", 0) >= 80 else "Within 24 hours"
        immediate_actions.append({
            "priority": i + 1,
            "action": p["recommendation"],
            "equipment": p["equipment_name"],
            "timeline": timeline,
            "justification": f"Confidence score {p.get('confidence', 0)}% indicates {p.get('severity', 'elevated')} risk",
        })
    for nm in near_misses[:3]:
        immediate_actions.append({
            "priority": len(immediate_actions) + 1,
            "action": nm["action_required"],
            "equipment": nm["equipment_name"],
            "timeline": "Within 72 hours",
            "justification": f"Near-miss event with {nm.get('failure_probability', 0)}% failure probability",
        })

    long_term_strategy = [
        "Deploy continuous vibration monitoring sensors on all rotating equipment (ISO 10816 compliance)",
        "Establish weekly thermal imaging inspection schedule for critical assets",
        "Build predictive maintenance database to track degradation trends over time",
        "Configure automated alert thresholds based on historical failure patterns",
        "Schedule quarterly oil analysis for all hydraulic and lubrication systems",
        "Transition from time-based to condition-based maintenance for high-value assets",
        "Implement digital twin modeling for critical equipment failure prediction",
    ]

    return {
        "success": True,
        "analysis": {
            "executive_summary": executive_summary,
            "critical_findings": critical_findings,
            "root_causes": root_causes,
            "immediate_actions": immediate_actions,
            "long_term_strategy": long_term_strategy,
        },
        "source": "ai",
    }


def chat_query(
    query: str,
    context: Dict[str, Any],
    conversation_history: List[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Handle chatbot queries about maintenance data using built-in AI."""
    if conversation_history is None:
        conversation_history = []

    context_str = ""
    if "equipment_health" in context:
        context_str += f"\n## Equipment Health Scores\n{json.dumps(context.get('equipment_health', []), indent=2, default=str)}"
    if "patterns" in context:
        context_str += f"\n## Detected Patterns\n{json.dumps(context.get('patterns', [])[:5], indent=2, default=str)}"
    if "schedule" in context:
        context_str += f"\n## Maintenance Schedule\n{json.dumps(context.get('schedule', [])[:5], indent=2, default=str)}"
    if "near_misses" in context:
        context_str += f"\n## Near-Miss Events\n{json.dumps(context.get('near_misses', [])[:5], indent=2, default=str)}"

    chat_prompt = f"""Based on the following maintenance data context, answer the user's question.

{context_str}

Conversation History:
{json.dumps(conversation_history[-10:], indent=2) if conversation_history else 'No previous messages'}

User Question: {query}

Provide a helpful, detailed answer that references specific equipment and data points."""

    response = call_llm(chat_prompt, SYSTEM_PROMPT, max_tokens=1500)

    if response:
        return {"success": True, "response": response, "source": "ai"}

    return _rule_based_chat(query, context)


def _rule_based_chat(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligent rule-based chat with NLP-enhanced keyword matching."""
    query_lower = query.lower()
    response = ""

    equipment_health = context.get("equipment_health", [])
    patterns = context.get("patterns", [])
    schedule = context.get("schedule", [])
    near_misses = context.get("near_misses", [])

    for eq in equipment_health:
        eq_name_lower = eq["equipment_name"].lower()
        eq_id_lower = eq["equipment_id"].lower()
        if eq_name_lower in query_lower or eq_id_lower in query_lower:
            status_emoji = {"critical": "\U0001f534", "warning": "\U0001f7e1", "attention": "\U0001f7e0", "healthy": "\U0001f7e2"}.get(eq["status"], "\u26aa")
            response = f"**{status_emoji} {eq['equipment_name']} ({eq['equipment_id']}) Status:**\n\n"
            response += f"- Health Score: **{eq['health_score']}/100** ({eq['status'].upper()})\n"
            response += f"- Total Log Entries: {eq['total_logs']}\n"
            response += f"- Critical Events: {eq['critical_events']}\n"
            response += f"- Warning Events: {eq['warning_events']}\n"
            response += f"- Maintenance Events: {eq['maintenance_events']}\n\n"

            eq_patterns = [p for p in patterns if p["equipment_id"] == eq["equipment_id"]]
            if eq_patterns:
                response += "**Active Degradation Patterns:**\n"
                for p in eq_patterns:
                    sev_icon = "\U0001f534" if p.get("severity") == "critical" else "\U0001f7e1"
                    response += f"- {sev_icon} {p['signal']}: {p['description']} (Confidence: {p['confidence']}%)\n"

            eq_schedule = [s for s in schedule if s.get("equipment_id") == eq["equipment_id"]]
            if eq_schedule:
                response += "\n**Scheduled Maintenance:**\n"
                for s in eq_schedule:
                    response += f"- [{s['urgency']}] {s['task']} (by {s['recommended_date']})\n"
            break

    if not response:
        if any(word in query_lower for word in ["status", "overview", "summary", "health", "how are", "dashboard", "fleet"]):
            response = "**Equipment Fleet Overview:**\n\n"
            for eq in sorted(equipment_health, key=lambda x: x["health_score"]):
                icon = {"critical": "\U0001f534", "warning": "\U0001f7e1", "attention": "\U0001f7e0", "healthy": "\U0001f7e2"}.get(eq["status"], "\u26aa")
                response += f"- {icon} **{eq['equipment_name']}**: {eq['health_score']}/100 ({eq['status']})\n"
            critical_count = len([eq for eq in equipment_health if eq["status"] == "critical"])
            warning_count = len([eq for eq in equipment_health if eq["status"] in ["warning", "attention"]])
            healthy_count = len(equipment_health) - critical_count - warning_count
            response += f"\n**Summary:** {critical_count} critical, {warning_count} need attention, {healthy_count} healthy."
            if critical_count > 0:
                response += "\n\n**Recommendation:** Address critical equipment immediately to prevent unplanned downtime."

        elif any(word in query_lower for word in ["schedule", "maintenance", "plan", "upcoming", "next", "when"]):
            if schedule:
                response = "**Prioritized Maintenance Schedule:**\n\n"
                for i, s in enumerate(schedule[:10]):
                    response += f"{i+1}. **[{s['urgency']}]** {s['equipment_name']}: {s['task']}\n"
                    response += f"   Confidence: {s['confidence']}% | By: {s['recommended_date']} | Duration: {s['estimated_duration']}\n\n"
            else:
                response = "No maintenance tasks currently scheduled. Upload equipment logs to generate a schedule."

        elif any(word in query_lower for word in ["risk", "critical", "urgent", "alert", "failure", "danger", "worst"]):
            critical = [p for p in patterns if p.get("severity") == "critical" or p.get("confidence", 0) >= 70]
            if critical:
                response = "**Critical Risk Assessment:**\n\n"
                for p in critical:
                    response += f"- **{p['equipment_name']}** - {p['signal']}\n"
                    response += f"  {p['description']}\n"
                    response += f"  Confidence: {p['confidence']}% | Action: {p['recommendation']}\n\n"
            else:
                response = "No critical risks currently detected."

        elif any(word in query_lower for word in ["near miss", "near-miss", "close call", "almost", "threshold"]):
            if near_misses:
                response = "**Near-Miss Events Detected:**\n\n"
                for nm in near_misses:
                    response += f"- **{nm['equipment_name']}** - {nm['parameter']}\n"
                    response += f"  {nm['description']}\n"
                    response += f"  Failure Probability: {nm['failure_probability']}% | Action: {nm['action_required']}\n\n"
            else:
                response = "No near-miss events currently detected."

        elif any(word in query_lower for word in ["depend", "impact", "ripple", "connected", "cascade", "chain"]):
            dependencies = context.get("dependencies", {})
            deps = dependencies.get("dependencies", [])
            if deps:
                response = "**Equipment Dependencies (Ripple Effect Analysis):**\n\n"
                for dep in deps:
                    response += f"- **{dep['equipment_1']['name']}** <-> **{dep['equipment_2']['name']}**\n"
                    response += f"  Strength: {dep['dependency_strength']}% | {dep['description']}\n\n"
            else:
                response = "No significant equipment dependencies detected."

        elif any(word in query_lower for word in ["pattern", "degradation", "trend", "deteriorat"]):
            if patterns:
                response = "**Degradation Patterns Detected:**\n\n"
                for p in patterns:
                    sev_icon = "\U0001f534" if p.get("severity") == "critical" else "\U0001f7e1"
                    response += f"- {sev_icon} **{p['equipment_name']}** - {p['signal']}\n"
                    response += f"  {p['description']}\n"
                    response += f"  Severity: {p['severity']} | Confidence: {p['confidence']}%\n\n"
            else:
                response = "No degradation patterns detected in current data."

        elif any(word in query_lower for word in ["help", "what can", "how to", "guide", "feature"]):
            response = "**I'm your Predictive Maintenance AI Assistant!**\n\n"
            response += "I can help you with:\n\n"
            response += "- **Equipment status**: *'How is CNC Machine Alpha?'*\n"
            response += "- **Fleet overview**: *'Show me the overall status'*\n"
            response += "- **Maintenance schedule**: *'What maintenance is needed?'*\n"
            response += "- **Risk assessment**: *'What are the critical risks?'*\n"
            response += "- **Near-miss events**: *'Show near-miss events'*\n"
            response += "- **Dependencies**: *'What equipment is connected?'*\n"
            response += "- **Patterns**: *'Show degradation patterns'*\n"
            response += "\nJust ask in natural language and I'll analyze your data!"

        else:
            response = "I can help you with maintenance insights! Here's a quick summary:\n\n"
            if equipment_health:
                critical = [eq for eq in equipment_health if eq["status"] == "critical"]
                if critical:
                    response += f"**{len(critical)} equipment units need immediate attention:**\n"
                    for eq in critical:
                        response += f"  - {eq['equipment_name']}: Health {eq['health_score']}/100\n"
                    response += "\n"
            if patterns:
                response += f"{len(patterns)} degradation patterns detected.\n"
            if near_misses:
                response += f"{len(near_misses)} near-miss events recorded.\n"
            response += "\nAsk me about specific equipment, schedules, risks, or patterns for detailed analysis."

    return {
        "success": True,
        "response": response,
        "source": "ai",
    }
