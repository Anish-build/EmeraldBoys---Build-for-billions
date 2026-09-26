import os
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

<<<<<<< HEAD
from schemas import MLPayload, FinalAgentResponse
=======
import uuid
from datetime import datetime, timezone
from schemas import MLPayload, FinalAgentResponse, RepairRequest, RepairResponse
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
from agent import evaluate_case
from database import init_db, get_db
import repositories

# Configure logging
logger = logging.getLogger("EmeraldAgentAPI")

# Ensure DB is initialized at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

# Also initialize immediately on module load for safety
init_db()

app = FastAPI(
    title="Emerald Boys AI Agent API",
    version="3.0",
    description="Evaluate a handpump ML diagnostic result using the Emerald Boys AI Agent with persistent case memory.",
    lifespan=lifespan
)

# Optional Configurable CORS (disabled if CORS_ORIGINS is not set)
cors_origins_env = os.getenv("CORS_ORIGINS")
if cors_origins_env and cors_origins_env.strip():
    origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
<<<<<<< HEAD
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

=======
else:
    origins = []
# Ensure Streamlit frontend can access
if "http://localhost:8501" not in origins:
    origins.append("http://localhost:8501")
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
@app.get(
    "/",
    summary="Root Service Status",
    description="Returns basic service information indicating the Agent API is running."
)
def read_root() -> Dict[str, str]:
    return {
        "service": "Emerald Boys AI Agent",
        "version": "2.0",
        "status": "running"
    }

@app.get(
    "/health",
    summary="Health Check",
    description="Performs a lightweight service health check without invoking LLM operations."
)
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "emerald-boys-agent",
        "version": "2.0"
    }

@app.post(
    "/evaluate",
    response_model=FinalAgentResponse,
    summary="Evaluate ML Diagnostic Payload",
    description="Evaluate a handpump ML diagnostic result, persist the diagnostic case and transition history, and return structured decision."
)
def evaluate_endpoint(payload: MLPayload, db: Session = Depends(get_db)):
    logger.info(f"Received evaluation request: {payload.case_id} for pump: {payload.pump_id}")
    
    try:
        # 1. Ensure pump record exists in database
        repositories.get_or_create_pump(
            db=db,
            pump_id=payload.pump_id,
            status=payload.current_state
        )

        # 2. Execute neuro-symbolic agent evaluation
        logger.info(f"Invoking evaluate_case for {payload.case_id}")
        response: FinalAgentResponse = evaluate_case(payload)
        logger.info(f"Agent evaluation completed for {payload.case_id}")

        # 3. Detect whether an illegal transition was overridden
        was_overridden = "[OVERRIDE]" in (response.audit_notes or "")

        # 4. Persist diagnostic case
        repositories.save_diagnostic_case(
            db=db,
            case_id=payload.case_id,
            pump_id=payload.pump_id,
            ml_prediction=payload.ml_prediction,
            ml_confidence=payload.ml_confidence,
            confidence_flag=response.confidence_flag,
            input_state=payload.current_state,
            proposed_state=response.proposed_state,
            final_state=response.proposed_state,
            action=response.action,
            needs_human_review=response.needs_human_review,
            explanation=response.explanation,
            audit_notes=response.audit_notes
        )

        # 5. Persist state transition history
        repositories.record_state_transition(
            db=db,
            case_id=payload.case_id,
            pump_id=payload.pump_id,
            from_state=payload.current_state,
            proposed_state=response.proposed_state,
            final_state=response.proposed_state,
            action=response.action,
            reason=response.explanation,
            was_overridden=was_overridden
        )

        # 6. Update pump status
        repositories.update_pump_status(
            db=db,
            pump_id=payload.pump_id,
            new_status=response.proposed_state
        )

        # 7. Generate escalation event if final validated state is ESCALATED
        if response.proposed_state == "ESCALATED":
            from events import record_escalation_event
            record_escalation_event(
                pump_id=payload.pump_id,
                case_id=payload.case_id,
                reason=response.explanation or "Case resulted in ESCALATED state.",
                final_state=response.proposed_state
            )

        logger.info(f"Successfully persisted case {payload.case_id} and transition for pump {payload.pump_id}")
        return response

    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Error evaluating and persisting case {payload.case_id}: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Agent evaluation failed",
                "detail": "An internal error occurred during agent evaluation or persistence.",
                "case_id": payload.case_id
            }
        )


@app.get(
    "/pumps/{pump_id}/history",
    summary="Get Pump Historical Cases and Transitions",
    description="Retrieve historical diagnostic cases and state transitions for a specific pump."
)
def get_pump_history_endpoint(pump_id: str, db: Session = Depends(get_db)):
    pump = repositories.get_pump_by_id(db, pump_id)
    if not pump:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Pump not found", "pump_id": pump_id}
        )
    cases = repositories.get_cases_for_pump(db, pump_id)
    transitions = repositories.get_transitions_for_pump(db, pump_id)
    from events import get_events_for_pump
    events_list = get_events_for_pump(pump_id)

    return {
        "pump_id": pump.pump_id,
        "current_status": pump.status,
        "location_info": pump.location_info,
        "cases_count": len(cases),
        "cases": [
            {
                "case_id": c.case_id,
                "ml_prediction": c.ml_prediction,
                "ml_confidence": c.ml_confidence,
                "confidence_flag": c.confidence_flag,
                "final_state": c.final_state,
                "action": c.action,
                "created_at": c.created_at.isoformat() if c.created_at else None
            }
            for c in cases
        ],
        "transitions": [
            {
                "case_id": t.case_id,
                "from_state": t.from_state,
                "proposed_state": t.proposed_state,
                "final_state": t.final_state,
                "action": t.action,
                "was_overridden": t.was_overridden,
                "created_at": t.created_at.isoformat() if t.created_at else None
            }
            for t in transitions
        ],
        "escalation_events": events_list
    }
<<<<<<< HEAD
=======


@app.post(
    "/repair/start",
    response_model=RepairResponse,
    summary="Start Pump Repair",
    description="Initiate repair on a pump currently in MAINTENANCE_REQUIRED status, transitioning to REPAIR_IN_PROGRESS."
)
def start_repair_endpoint(req: RepairRequest, db: Session = Depends(get_db)):
    pump = repositories.get_pump_by_id(db, req.pump_id)
    if not pump:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Pump not found", "pump_id": req.pump_id}
        )
    
    if pump.status != "MAINTENANCE_REQUIRED":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Illegal transition request",
                "detail": f"Cannot start repair on pump '{pump.pump_id}' in state '{pump.status}'. Pump must be in 'MAINTENANCE_REQUIRED' state.",
                "pump_id": pump.pump_id,
                "current_state": pump.status
            }
        )
    
    case_id = f"REPAIR-START-{uuid.uuid4().hex[:8]}"
    prev_state = pump.status
    next_state = "REPAIR_IN_PROGRESS"

    transition = repositories.record_state_transition(
        db=db,
        case_id=case_id,
        pump_id=pump.pump_id,
        from_state=prev_state,
        proposed_state=next_state,
        final_state=next_state,
        action="START_REPAIR",
        reason=req.notes or f"Repair initiated by technician {req.technician_id or 'DISPATCHED'}.",
        was_overridden=False
    )
    repositories.update_pump_status(db, pump.pump_id, next_state)

    return RepairResponse(
        pump_id=pump.pump_id,
        previous_state=prev_state,
        current_state=next_state,
        action="START_REPAIR",
        case_id=case_id,
        message="Repair procedure successfully initiated.",
        timestamp=transition.created_at.isoformat() if transition.created_at else datetime.now(timezone.utc).isoformat()
    )


@app.post(
    "/repair/complete",
    response_model=RepairResponse,
    summary="Complete Pump Repair",
    description="Mark repair complete on a pump currently in REPAIR_IN_PROGRESS status, transitioning to VERIFICATION_PENDING."
)
def complete_repair_endpoint(req: RepairRequest, db: Session = Depends(get_db)):
    pump = repositories.get_pump_by_id(db, req.pump_id)
    if not pump:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Pump not found", "pump_id": req.pump_id}
        )
    
    if pump.status != "REPAIR_IN_PROGRESS":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Illegal transition request",
                "detail": f"Cannot complete repair on pump '{pump.pump_id}' in state '{pump.status}'. Pump must be in 'REPAIR_IN_PROGRESS' state.",
                "pump_id": pump.pump_id,
                "current_state": pump.status
            }
        )
    
    case_id = f"REPAIR-COMPLETE-{uuid.uuid4().hex[:8]}"
    prev_state = pump.status
    next_state = "VERIFICATION_PENDING"

    transition = repositories.record_state_transition(
        db=db,
        case_id=case_id,
        pump_id=pump.pump_id,
        from_state=prev_state,
        proposed_state=next_state,
        final_state=next_state,
        action="COMPLETE_REPAIR",
        reason=req.notes or "Physical repair work completed; awaiting acoustic verification.",
        was_overridden=False
    )
    repositories.update_pump_status(db, pump.pump_id, next_state)

    return RepairResponse(
        pump_id=pump.pump_id,
        previous_state=prev_state,
        current_state=next_state,
        action="COMPLETE_REPAIR",
        case_id=case_id,
        message="Repair completed. Pump is now awaiting post-repair verification.",
        timestamp=transition.created_at.isoformat() if transition.created_at else datetime.now(timezone.utc).isoformat()
    )


@app.post(
    "/verify",
    response_model=FinalAgentResponse,
    summary="Post-Repair Acoustic Verification",
    description="Ingest verification MLPayload, evaluate post-repair pump health, enforce deterministic state validation, and persist before/after history."
)
def verify_endpoint(payload: MLPayload, db: Session = Depends(get_db)):
    logger.info(f"Received verification request for case {payload.case_id}, pump: {payload.pump_id}")
    
    # 1. Validate pump exists in DB
    pump = repositories.get_pump_by_id(db, payload.pump_id)
    if not pump:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Pump not found", "pump_id": payload.pump_id}
        )

    # 2. Confirm pump is currently in VERIFICATION_PENDING
    if pump.status != "VERIFICATION_PENDING":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Illegal verification state",
                "detail": f"Pump '{pump.pump_id}' is in status '{pump.status}'. Verification requires 'VERIFICATION_PENDING'.",
                "pump_id": pump.pump_id,
                "current_state": pump.status
            }
        )
    
    try:
        # 3. Agent evaluates the verification case
        agent_response = evaluate_case(payload)

        # 4. Backend Authority: Deterministic verification state determination
        # Low confidence (< 0.40) -> ESCALATED
        # High/Medium confidence NORMAL -> HEALTHY
        # High/Medium confidence ABNORMAL -> MAINTENANCE_REQUIRED (repair unsuccessful)
        if payload.ml_confidence < 0.40:
            final_state = "ESCALATED"
            action = "ESCALATE"
            needs_human_review = True
            verification_note = "Post-repair verification confidence is LOW (< 0.40). Escalated for manual technician inspection."
        elif payload.ml_prediction == "NORMAL":
            final_state = "HEALTHY"
            action = "PROPOSE_TRANSITION"
            needs_human_review = False
            verification_note = "Post-repair acoustic verification confirmed NORMAL pump operation. Transitioned to HEALTHY."
        else: # ABNORMAL
            final_state = "MAINTENANCE_REQUIRED"
            action = "PROPOSE_TRANSITION"
            needs_human_review = True
            verification_note = "Post-repair acoustic verification detected persistent ABNORMAL operation. Returned to MAINTENANCE_REQUIRED."

        was_overridden = final_state != agent_response.proposed_state
        audit_notes = agent_response.audit_notes + f" [POST-REPAIR VERIFICATION: {verification_note}]"
        explanation = agent_response.explanation

        # 5. Persist the verification DiagnosticCase in SQLite
        repositories.save_diagnostic_case(
            db=db,
            case_id=payload.case_id,
            pump_id=payload.pump_id,
            ml_prediction=payload.ml_prediction,
            ml_confidence=payload.ml_confidence,
            confidence_flag=agent_response.confidence_flag,
            input_state="VERIFICATION_PENDING",
            proposed_state=final_state,
            final_state=final_state,
            action=action,
            needs_human_review=needs_human_review,
            explanation=explanation,
            audit_notes=audit_notes
        )

        # 6. Persist StateTransition
        repositories.record_state_transition(
            db=db,
            case_id=payload.case_id,
            pump_id=payload.pump_id,
            from_state="VERIFICATION_PENDING",
            proposed_state=final_state,
            final_state=final_state,
            action=action,
            reason=verification_note,
            was_overridden=was_overridden
        )

        # 7. Update pump status
        repositories.update_pump_status(
            db=db,
            pump_id=payload.pump_id,
            new_status=final_state
        )

        # 8. Trigger escalation event if final state is ESCALATED
        if final_state == "ESCALATED":
            from events import record_escalation_event
            record_escalation_event(
                pump_id=payload.pump_id,
                case_id=payload.case_id,
                reason=verification_note,
                final_state="ESCALATED"
            )

        logger.info(f"Successfully processed post-repair verification for {payload.pump_id}: {final_state}")

        report = agent_response.diagnostic_report
        if report:
            report.next_action = f"Transition to {final_state}"

        return FinalAgentResponse(
            case_id=payload.case_id,
            pump_id=payload.pump_id,
            action=action,
            current_state="VERIFICATION_PENDING",
            proposed_state=final_state,
            ml_prediction=payload.ml_prediction,
            ml_confidence=payload.ml_confidence,
            confidence_flag=agent_response.confidence_flag,
            needs_human_review=needs_human_review,
            explanation=explanation,
            audit_notes=audit_notes,
            diagnostic_report=report
        )

    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.error(f"Error during post-repair verification for {payload.case_id}: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Verification failed",
                "detail": "An internal error occurred during post-repair verification evaluation or persistence.",
                "case_id": payload.case_id
            }
        )
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
