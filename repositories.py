import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from database import Pump, DiagnosticCase, StateTransition

logger = logging.getLogger("EmeraldRepositories")


def get_or_create_pump(
    db: Session,
    pump_id: str,
    status: str = "DIAGNOSIS_PENDING",
    location_info: Optional[str] = None
) -> Pump:
    """Retrieves an existing pump or creates a new entry if not found."""
    pump = db.query(Pump).filter(Pump.pump_id == pump_id).first()
    if not pump:
        pump = Pump(
            pump_id=pump_id,
            status=status,
            location_info=location_info
        )
        db.add(pump)
        db.commit()
        db.refresh(pump)
        logger.info(f"Created new pump record: {pump_id}")
    return pump


def update_pump_status(db: Session, pump_id: str, new_status: str) -> Optional[Pump]:
    """Updates the current status of an existing pump."""
    pump = db.query(Pump).filter(Pump.pump_id == pump_id).first()
    if pump:
        pump.status = new_status
        db.commit()
        db.refresh(pump)
        logger.info(f"Updated pump {pump_id} status to {new_status}")
    return pump


def get_pump_by_id(db: Session, pump_id: str) -> Optional[Pump]:
    """Finds a pump record by pump_id."""
    return db.query(Pump).filter(Pump.pump_id == pump_id).first()


def save_diagnostic_case(
    db: Session,
    case_id: str,
    pump_id: str,
    ml_prediction: str,
    ml_confidence: float,
    confidence_flag: str,
    input_state: str,
    proposed_state: str,
    final_state: str,
    action: str,
    needs_human_review: bool,
    explanation: str,
    audit_notes: str
) -> DiagnosticCase:
    """Persists a new diagnostic case evaluation record."""
    diag_case = DiagnosticCase(
        case_id=case_id,
        pump_id=pump_id,
        ml_prediction=ml_prediction,
        ml_confidence=ml_confidence,
        confidence_flag=confidence_flag,
        input_state=input_state,
        proposed_state=proposed_state,
        final_state=final_state,
        action=action,
        needs_human_review=needs_human_review,
        explanation=explanation,
        audit_notes=audit_notes
    )
    db.add(diag_case)
    db.commit()
    db.refresh(diag_case)
    logger.info(f"Saved diagnostic case {case_id} for pump {pump_id}")
    return diag_case


def record_state_transition(
    db: Session,
    case_id: str,
    pump_id: str,
    from_state: str,
    proposed_state: str,
    final_state: str,
    action: str,
    reason: Optional[str] = None,
    was_overridden: bool = False
) -> StateTransition:
    """Appends an immutable state transition record to the audit history."""
    transition = StateTransition(
        case_id=case_id,
        pump_id=pump_id,
        from_state=from_state,
        proposed_state=proposed_state,
        final_state=final_state,
        action=action,
        reason=reason,
        was_overridden=was_overridden
    )
    db.add(transition)
    db.commit()
    db.refresh(transition)
    logger.info(
        f"Recorded transition for {pump_id} ({case_id}): {from_state} -> {final_state} (Overridden={was_overridden})"
    )
    return transition


def get_case_by_id(db: Session, case_id: str) -> Optional[DiagnosticCase]:
    """Retrieves a diagnostic case by case_id."""
    return db.query(DiagnosticCase).filter(DiagnosticCase.case_id == case_id).order_by(DiagnosticCase.created_at.desc()).first()


def get_cases_for_pump(db: Session, pump_id: str, limit: int = 50) -> List[DiagnosticCase]:
    """Retrieves historical diagnostic cases for a pump, newest first."""
    return db.query(DiagnosticCase).filter(DiagnosticCase.pump_id == pump_id).order_by(DiagnosticCase.created_at.desc()).limit(limit).all()


def get_transitions_for_pump(db: Session, pump_id: str, limit: int = 50) -> List[StateTransition]:
    """Retrieves historical state transitions for a pump, newest first."""
    return db.query(StateTransition).filter(StateTransition.pump_id == pump_id).order_by(StateTransition.created_at.desc()).limit(limit).all()


def get_transitions_for_case(db: Session, case_id: str) -> List[StateTransition]:
    """Retrieves state transitions linked to a specific case_id."""
    return db.query(StateTransition).filter(StateTransition.case_id == case_id).order_by(StateTransition.created_at.asc()).all()
