import os
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from schemas import MLPayload, FinalAgentResponse
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
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

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
