import json
from typing import Dict, Any, List, Optional

def get_pump_context(pump_id: str) -> str:
    """Fetch baseline historical maintenance context and specs for a given pump_id."""
    if not isinstance(pump_id, str) or not pump_id.strip():
        return json.dumps({"error": "Invalid pump_id provided."})

    mock_db = {
        "PUMP-001": {
            "age_years": 8,
            "last_maintenance_date": "2023-01-15",
            "previous_failures": 3,
            "known_issues": "Prone to severe seal degradation."
        },
        "PUMP-002": {
            "age_years": 1,
            "last_maintenance_date": "2023-09-01",
            "previous_failures": 0,
            "known_issues": "None"
        }
    }
    
    data = mock_db.get(pump_id)
    if not data:
        return json.dumps({"error": "Pump not found in database."})
    
    return json.dumps(data)


def get_pump_history(pump_id: str) -> str:
    """
    Safely retrieves historical diagnostic cases, state transitions, and escalation events
    from SQLite through controlled repository functions.
    The Agent never executes SQL directly.
    """
    if not isinstance(pump_id, str) or not pump_id.strip():
        return json.dumps({"error": "Invalid pump_id provided."})

    try:
        from database import SessionLocal
        import repositories
        from events import get_events_for_pump

        db = SessionLocal()
        try:
            cases = repositories.get_cases_for_pump(db, pump_id, limit=10)
            transitions = repositories.get_transitions_for_pump(db, pump_id, limit=10)
            escalations = get_events_for_pump(pump_id)

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
                "pump_id": pump_id,
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
        return json.dumps({"error": f"Database context retrieval failed: {str(e)}"})


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
        "description": "Fetch baseline static maintenance specifications and hardware context for a specific handpump.",
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