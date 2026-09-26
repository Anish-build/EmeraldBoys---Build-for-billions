import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("EmeraldEvents")

# In-memory local event store for hackathon prototype
_LOCAL_EVENT_STORE: List[Dict[str, Any]] = []


def record_escalation_event(
    pump_id: str,
    case_id: str,
    reason: str,
    final_state: str = "ESCALATED"
) -> Optional[Dict[str, Any]]:
    """
    Generates and records an escalation event ONLY when final validated state is ESCALATED.
    Dispatches to optional ESCALATION_WEBHOOK_URL if configured, handling failures safely.
    """
    # Deterministic rule: Only generate escalation events when the FINAL validated state is ESCALATED.
    if final_state != "ESCALATED":
        logger.debug(f"Skipping escalation event: final_state '{final_state}' is not ESCALATED.")
        return None

    event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    event: Dict[str, Any] = {
        "event_id": event_id,
        "pump_id": pump_id,
        "case_id": case_id,
        "event_type": "PUMP_ESCALATION",
        "reason": reason,
        "final_state": final_state,
        "timestamp": timestamp,
        "webhook_status": "DISABLED"
    }

    # Record event in local event store
    _LOCAL_EVENT_STORE.append(event)
    logger.warning(
        f"ESCALATION EVENT [{event_id}]: Pump {pump_id}, Case {case_id}, Reason: {reason}"
    )

    # Optional Webhook Dispatch
    webhook_url = os.getenv("ESCALATION_WEBHOOK_URL")
    if webhook_url and webhook_url.strip():
        try:
            req = urllib.request.Request(
                url=webhook_url.strip(),
                data=json.dumps(event).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "EmeraldAgent/3.0"},
                method="POST"
            )
            # Safe network request with strict timeout
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                status_code = resp.getcode()
                event["webhook_status"] = f"SENT_{status_code}"
                logger.info(f"Escalation event {event_id} dispatched to webhook: {status_code}")
        except Exception as e:
            # Webhook failures must NEVER break or crash the diagnostic workflow
            event["webhook_status"] = f"FAILED: {str(e)}"
            logger.warning(
                f"Escalation webhook dispatch failed safely for event {event_id}: {str(e)}"
            )
    else:
        logger.debug("Escalation webhook not configured; event logged locally only.")

    return event


def get_escalation_events() -> List[Dict[str, Any]]:
    """Returns a copy of all recorded escalation events."""
    return list(_LOCAL_EVENT_STORE)


def get_events_for_pump(pump_id: str) -> List[Dict[str, Any]]:
    """Returns all escalation events for a given pump_id, newest first."""
    return [e for e in reversed(_LOCAL_EVENT_STORE) if e["pump_id"] == pump_id]


def clear_events() -> None:
    """Clears the in-memory event store (primarily for test isolation)."""
    _LOCAL_EVENT_STORE.clear()
