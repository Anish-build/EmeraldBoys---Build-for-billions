import os
import json
import time
from database import init_db, SessionLocal
import repositories
from ml_adapter import MLAdapter, predict, ADAPTER_MODE
from agent import evaluate_case
from tools import execute_tool_safely
from events import get_events_for_pump, clear_events

def run_v3_demo():
    print("=" * 80)
    print("      EMERALD BOYS — V3 COMPLETE LAPTOP DEMONSTRATION")
    print("      Rural Handpump Neuro-Symbolic Diagnostic & Memory Engine")
    print("=" * 80)
    print(f"ML Layer Status: [{ADAPTER_MODE}] (Deterministic Software Prototype)")
    print("Persistence:     Local SQLite (data/emerald_boys.db)")
    print("State Machine:   Authoritative Deterministic Python Layer")
    print("=" * 80)

    # 1. Initialize local SQLite database
    init_db()
    db = SessionLocal()

    demo_pump_id = "PUMP-V3-DEMO"
    repositories.get_or_create_pump(db, pump_id=demo_pump_id, status="HEALTHY", location_info="Kibera Cluster 4")
    print(f"\n[INIT] Pump {demo_pump_id} registered in SQLite. Initial status: HEALTHY")

    # Step 1: Process Audio Anomaly via ML Adapter
    print("\n" + "=" * 60)
    print("STAGE 1: Acoustic Pipeline & First Evaluation")
    print("=" * 60)
    audio_path = "samples/audio/seal_leak_anomaly.wav"
    print(f"1. Ingesting prerecorded acoustic sample: {audio_path}")
    
    # ML Adapter processes audio deterministically into MLPayload
    # Transitioning pump from HEALTHY -> DIAGNOSIS_PENDING
    repositories.update_pump_status(db, demo_pump_id, "DIAGNOSIS_PENDING")
    ml_payload = predict(
        audio_path=audio_path,
        pump_id=demo_pump_id,
        current_state="DIAGNOSIS_PENDING",
        case_id="CASE-DEMO-01"
    )
    print(f"2. ML Adapter produced MLPayload [{ADAPTER_MODE}]:")
    print(f"   - Case ID:         {ml_payload.case_id}")
    print(f"   - Pump ID:         {ml_payload.pump_id}")
    print(f"   - ML Prediction:   {ml_payload.ml_prediction}")
    print(f"   - ML Confidence:   {ml_payload.ml_confidence:.2f}")

    print("\n3. Invoking AI Agent with Safe Tool Dispatching...")
    t0 = time.time()
    response = evaluate_case(ml_payload)
    elapsed = time.time() - t0
    print(f"   Decision generated in {elapsed:.2f}s:")
    print(f"   - Proposed State:    {response.proposed_state}")
    print(f"   - Action:            {response.action}")
    print(f"   - Confidence Flag:   {response.confidence_flag}")
    print(f"   - Needs Human Review:{response.needs_human_review}")
    print(f"   - Explanation:       {response.explanation}")

    # Persist Case 1 in SQLite
    repositories.save_diagnostic_case(
        db=db,
        case_id=response.case_id,
        pump_id=demo_pump_id,
        ml_prediction=response.ml_prediction,
        ml_confidence=response.ml_confidence,
        confidence_flag=response.confidence_flag,
        input_state=response.current_state,
        proposed_state=response.proposed_state,
        final_state=response.proposed_state,
        action=response.action,
        needs_human_review=response.needs_human_review,
        explanation=response.explanation,
        audit_notes=response.audit_notes
    )
    repositories.record_state_transition(
        db=db,
        case_id=response.case_id,
        pump_id=demo_pump_id,
        from_state=response.current_state,
        proposed_state=response.proposed_state,
        final_state=response.proposed_state,
        action=response.action,
        reason=response.explanation,
        was_overridden=False
    )
    repositories.update_pump_status(db, demo_pump_id, response.proposed_state)
    print(f"4. Persisted Case 1 to SQLite. Updated Pump status -> {response.proposed_state}")

    # Stage 2: Controlled Agent Historical Retrieval
    print("\n" + "=" * 60)
    print("STAGE 2: Historical Database Retrieval through Agent Tool")
    print("=" * 60)
    history_json = execute_tool_safely("get_pump_history", {"pump_id": demo_pump_id})
    history = json.loads(history_json)
    print(f"Agent retrieved persistent memory for {demo_pump_id}:")
    print(f"   - Previous Cases:    {history.get('previous_cases')}")
    print(f"   - Recent States:     {history.get('recent_states')}")
    print(f"   - Past Anomalies:    {history.get('previous_anomalies')}")

    # Stage 3: Low-Confidence Anomaly & Escalation Event
    print("\n" + "=" * 60)
    print("STAGE 3: Low-Confidence Anomaly & Deterministic Escalation")
    print("=" * 60)
    low_conf_audio = "samples/audio/low_confidence_chatter.wav"
    print(f"1. Ingesting noisy acoustic sample: {low_conf_audio}")
    ml_payload_2 = predict(
        audio_path=low_conf_audio,
        pump_id=demo_pump_id,
        current_state="MAINTENANCE_REQUIRED",
        case_id="CASE-DEMO-02"
    )
    print(f"2. ML Adapter produced MLPayload [{ADAPTER_MODE}]:")
    print(f"   - Prediction: {ml_payload_2.ml_prediction}, Confidence: {ml_payload_2.ml_confidence:.2f}")

    print("\n3. Evaluating Case 2 with history context...")
    response_2 = evaluate_case(ml_payload_2)
    print(f"   - Final Validated State: {response_2.proposed_state}")
    print(f"   - Action:                {response_2.action}")
    print(f"   - Needs Human Review:    {response_2.needs_human_review}")

    # Persist Case 2 & Trigger Escalation if ESCALATED
    repositories.save_diagnostic_case(
        db=db,
        case_id=response_2.case_id,
        pump_id=demo_pump_id,
        ml_prediction=response_2.ml_prediction,
        ml_confidence=response_2.ml_confidence,
        confidence_flag=response_2.confidence_flag,
        input_state=response_2.current_state,
        proposed_state=response_2.proposed_state,
        final_state=response_2.proposed_state,
        action=response_2.action,
        needs_human_review=response_2.needs_human_review,
        explanation=response_2.explanation,
        audit_notes=response_2.audit_notes
    )
    repositories.record_state_transition(
        db=db,
        case_id=response_2.case_id,
        pump_id=demo_pump_id,
        from_state=response_2.current_state,
        proposed_state=response_2.proposed_state,
        final_state=response_2.proposed_state,
        action=response_2.action,
        reason=response_2.explanation,
        was_overridden="[OVERRIDE]" in (response_2.audit_notes or "")
    )
    repositories.update_pump_status(db, demo_pump_id, response_2.proposed_state)

    if response_2.proposed_state == "ESCALATED":
        from events import record_escalation_event
        event = record_escalation_event(
            pump_id=demo_pump_id,
            case_id=response_2.case_id,
            reason=response_2.explanation,
            final_state="ESCALATED"
        )
        print(f"\n4. [ESCALATION EVENT GENERATED]:")
        print(f"   - Event ID:       {event['event_id']}")
        print(f"   - Event Type:     {event['event_type']}")
        print(f"   - Timestamp:      {event['timestamp']}")
        print(f"   - Webhook Status: {event['webhook_status']}")

    # Final summary
    pump_record = repositories.get_pump_by_id(db, demo_pump_id)
    all_cases = repositories.get_cases_for_pump(db, demo_pump_id)
    all_transitions = repositories.get_transitions_for_pump(db, demo_pump_id)
    db.close()

    print("\n" + "=" * 80)
    print("V3 DEMO COMPLETE — SYSTEM STATE SUMMARY:")
    print(f"Pump ID:            {pump_record.pump_id}")
    print(f"Final Status:       {pump_record.status}")
    print(f"Total Cases Saved:  {len(all_cases)}")
    print(f"Total Transitions:  {len(all_transitions)}")
    print(f"Escalation Events:  {len(get_events_for_pump(demo_pump_id))}")
    print("=" * 80)

if __name__ == "__main__":
    run_v3_demo()
