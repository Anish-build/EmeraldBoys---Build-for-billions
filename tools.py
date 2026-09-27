import json
import logging
from typing import Dict, Any, List, Optional
from database import SessionLocal
import repositories

logger = logging.getLogger("EmeraldTools")


def get_pump_context(pump_id: str) -> str:
    """
    Fetch baseline static maintenance context and hardware specs for a given pump_id
    authoritatively from the SQLite database.
    """
    if not isinstance(pump_id, str) or not pump_id.strip():
        return json.dumps({"error": "Invalid pump_id provided."})

    cleaned_id = pump_id.strip()
    db = SessionLocal()
    try:
        pump = repositories.get_pump_by_id(db, cleaned_id)
        if not pump or (pump.location_info and "Unregistered" in pump.location_info):
            return json.dumps({"error": "Pump not found in database."})
        
        data = {
            "pump_id": pump.pump_id,
            "status": pump.status,
            "location_info": pump.location_info,
            "age_years": pump.age_years,
            "last_maintenance_date": pump.last_maintenance_date,
            "previous_failures": pump.previous_failures,
            "known_issues": pump.known_issues
        }
        return json.dumps(data)
    except Exception as e:
        logger.error(f"Error querying pump context for {cleaned_id}: {e}")
        return json.dumps({"error": f"Database context query error: {str(e)}"})
    finally:
        db.close()


def get_pump_history(pump_id: str) -> str:
    """
    Safely retrieves historical diagnostic cases, state transitions, and escalation events
    from SQLite through controlled repository functions.
    The Agent never executes SQL directly.
    """
    if not isinstance(pump_id, str) or not pump_id.strip():
        return json.dumps({"error": "Invalid pump_id provided."})

    cleaned_id = pump_id.strip()
    try:
        from events import get_events_for_pump

        db = SessionLocal()
        try:
            cases = repositories.get_cases_for_pump(db, cleaned_id, limit=10)
            transitions = repositories.get_transitions_for_pump(db, cleaned_id, limit=10)
            escalations = get_events_for_pump(cleaned_id)

            recent_states = [t.final_state for t in transitions]
            previous_anomalies = [c.ml_prediction for c in cases if c.ml_prediction == "ABNORMAL"]

            cases_detail = [
                {
                    "case_id": c.case_id,
                    "ml_prediction": c.ml_prediction,
                    "ml_confidence": c.ml_confidence,
                    "confidence_flag": c.confidence_flag,
                    "final_state": c.final_state,
                    "action": c.action,
                    "was_overridden": c.was_overridden,
                    "timestamp": c.created_at.isoformat() if c.created_at else None
                }
                for c in cases
            ]

            transitions_detail = [
                {
                    "case_id": t.case_id,
                    "from_state": t.from_state,
                    "final_state": t.final_state,
                    "action": t.action,
                    "was_overridden": t.was_overridden,
                    "timestamp": t.created_at.isoformat() if t.created_at else None
                }
                for t in transitions
            ]

            escalations_detail = [
                {
                    "event_id": e.get("event_id"),
                    "case_id": e.get("case_id"),
                    "reason": e.get("reason"),
                    "timestamp": e.get("timestamp")
                }
                for e in escalations
            ]

            history_data = {
                "pump_id": cleaned_id,
                "previous_cases": len(cases),
                "recent_states": recent_states,
                "previous_anomalies": previous_anomalies,
                "last_audit_notes": cases[0].audit_notes if cases else None,
                "cases_detail": cases_detail,
                "transitions_detail": transitions_detail,
                "escalation_events": escalations_detail
            }
            return json.dumps(history_data)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error querying pump history for {cleaned_id}: {e}")
        return json.dumps({"error": f"Database history retrieval failed: {str(e)}"})


def execute_tool_safely(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Safely dispatches authorized tools with input validation and unknown-tool blocking."""
    if tool_name == "get_pump_context":
        pump_id = arguments.get("pump_id")
        if not pump_id:
            return json.dumps({"error": "Missing required argument 'pump_id'."})
        return get_pump_context(str(pump_id))
    elif tool_name == "get_pump_history":
        pump_id = arguments.get("pump_id")
        if not pump_id:
            return json.dumps({"error": "Missing required argument 'pump_id'."})
        return get_pump_history(str(pump_id))
    
    return json.dumps({"error": f"Unauthorized tool call attempted: {tool_name}"})


PUMP_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_pump_context",
        "description": "Fetch baseline static maintenance specifications and hardware context for a specific handpump from database memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pump_id": {
                    "type": "string",
                    "description": "The unique pump identifier (e.g., PUMP-001)"
                }
            },
            "required": ["pump_id"]
        }
    }
}

PUMP_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_pump_history",
        "description": "Fetch persistent historical diagnostic cases, state transitions, previous anomalies, and escalation events for a specific handpump from database memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pump_id": {
                    "type": "string",
                    "description": "The unique pump identifier (e.g., PUMP-001)"
                }
            },
            "required": ["pump_id"]
        }
    }
}