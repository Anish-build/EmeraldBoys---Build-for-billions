import json
from fastapi.testclient import TestClient
from api import app
from schemas import MLPayload, RepairRequest

client = TestClient(app)

PUMP_ID = "PUMP-001"

def print_section(title: str):
    print(f"\n{'='*10} {title} {'='*10}\n")

def evaluate_initial():
    payload = {
        "case_id": "CASE-INIT",
        "pump_id": PUMP_ID,
        "current_state": "DIAGNOSIS_PENDING",
        "ml_prediction": "ABNORMAL",
        "ml_confidence": 0.92,
    }
    resp = client.post("/evaluate", json=payload)
    print_section("Initial Evaluation Response")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def start_repair():
    req = {
        "pump_id": PUMP_ID,
        "notes": "Technician begins repair",
    }
    resp = client.post("/repair/start", json=req)
    print_section("Repair Start Response")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def complete_repair():
    req = {
        "pump_id": PUMP_ID,
        "notes": "Repair finished, pump back in service",
    }
    resp = client.post("/repair/complete", json=req)
    print_section("Repair Complete Response")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def verify_post_repair():
    # Simulate external ML payload after repair (e.g., normal rhythm)
    payload = {
        "case_id": "CASE-VERIF",
        "pump_id": PUMP_ID,
        "current_state": "VERIFICATION_PENDING",
        "ml_prediction": "NORMAL",
        "ml_confidence": 0.88,
    }
    resp = client.post("/verify", json=payload)
    print_section("Verification Response")
    print(json.dumps(resp.json(), indent=2))
    return resp.json()

def show_history():
    resp = client.get(f"/pumps/{PUMP_ID}/history")
    print_section("Pump History After Full Lifecycle")
    print(json.dumps(resp.json(), indent=2))

if __name__ == "__main__":
    evaluate_initial()
    start_repair()
    complete_repair()
    verify_post_repair()
    show_history()
