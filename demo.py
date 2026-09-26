import json
import time
from schemas import MLPayload
from agent import evaluate_case

demo_cases = [
    {
        "title": "CASE 1: High-Confidence Anomaly (Standard Maintenance Flow)",
        "payload": MLPayload(
            case_id="CASE-101",
            pump_id="PUMP-001",
            current_state="DIAGNOSIS_PENDING",
            ml_prediction="ABNORMAL",
            ml_confidence=0.92
        )
    },
    {
        "title": "CASE 2: Low-Confidence Anomaly (Uncertainty Handling)",
        "payload": MLPayload(
            case_id="CASE-102",
            pump_id="PUMP-002",
            current_state="DIAGNOSIS_PENDING",
            ml_prediction="ABNORMAL",
            ml_confidence=0.35
        )
    },
    {
        "title": "CASE 3: High-Confidence Normal (Return to Healthy State)",
        "payload": MLPayload(
            case_id="CASE-103",
            pump_id="PUMP-001",
            current_state="DIAGNOSIS_PENDING",
            ml_prediction="NORMAL",
            ml_confidence=0.88
        )
    },
    {
        "title": "CASE 4: Unknown Pump ID (Context Fetch Failure)",
        "payload": MLPayload(
            case_id="CASE-104",
            pump_id="PUMP-999",
            current_state="DIAGNOSIS_PENDING",
            ml_prediction="ABNORMAL",
            ml_confidence=0.85
        )
    },
    {
        "title": "CASE 5: Illegal Transition Proposal (Backend Override Safeguard)",
        "payload": MLPayload(
            case_id="CASE-105",
            pump_id="PUMP-001",
            current_state="HEALTHY",
            ml_prediction="ABNORMAL",
            ml_confidence=0.95
        )
    }
]

def run_demo():
    print("=" * 80)
    print("      EMERALD BOYS - NEURO-SYMBOLIC DIAGNOSTIC AGENT DEMO (V1)")
    print("=" * 80)

    for case in demo_cases:
        print(f"\n\n>>> {case['title']}")
        payload: MLPayload = case["payload"]
        
        print("-" * 60)
        print("1. UPSTREAM INPUT (ML Payload):")
        print(f"   - Case ID:         {payload.case_id}")
        print(f"   - Pump ID:         {payload.pump_id}")
        print(f"   - Current State:   {payload.current_state}")
        print(f"   - ML Prediction:   {payload.ml_prediction}")
        print(f"   - ML Confidence:   {payload.ml_confidence}")
        print("-" * 60)

        start_time = time.time()
        result = evaluate_case(payload)
        elapsed = time.time() - start_time

        print("\n2. AGENT OUTPUT & DETERMINISTIC VALIDATION:")
        print(json.dumps(result.model_dump(), indent=2))
        print(f"\n[Processed in {elapsed:.2f}s]")
        print("=" * 80)

if __name__ == "__main__":
    run_demo()