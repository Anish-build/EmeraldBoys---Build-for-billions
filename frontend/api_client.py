import os
import uuid
import requests
from typing import Dict, Any

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def _post(endpoint: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    resp = requests.post(url, json=json_data)
    resp.raise_for_status()
    return resp.json()

def _get(endpoint: str) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def evaluate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send MLPayload to /evaluate and return FinalAgentResponse."""
    return _post("/evaluate", payload)

def repair_start(pump_id: str, notes: str = "", technician_id: str = "") -> Dict[str, Any]:
    """Start repair via /repair/start."""
    data = {"pump_id": pump_id, "notes": notes, "technician_id": technician_id}
    return _post("/repair/start", data)

def repair_complete(pump_id: str, notes: str = "", technician_id: str = "") -> Dict[str, Any]:
    """Complete repair via /repair/complete."""
    data = {"pump_id": pump_id, "notes": notes, "technician_id": technician_id}
    return _post("/repair/complete", data)

def verify(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Post-repair verification via /verify."""
    return _post("/verify", payload)

def get_history(pump_id: str) -> Dict[str, Any]:
    """Retrieve pump history via /pumps/{pump_id}/history."""
    return _get(f"/pumps/{pump_id}/history")
