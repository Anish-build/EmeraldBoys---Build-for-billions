from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

PumpState = Literal[
    "HEALTHY", 
    "DIAGNOSIS_PENDING", 
    "MAINTENANCE_REQUIRED", 
    "REPAIR_IN_PROGRESS", 
    "VERIFICATION_PENDING", 
    "ESCALATED"
]

ActionType = Literal["PROPOSE_TRANSITION", "ESCALATE"]


class MLPayload(BaseModel):
    case_id: str = Field(description="Unique case evaluation identifier.")
    pump_id: str = Field(description="Unique pump identifier.")
    current_state: PumpState = Field(description="Current lifecycle state of the pump.")
    ml_prediction: str = Field(description="Predicted acoustic classification (e.g., NORMAL, ABNORMAL).")
    ml_confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="ML model probability score constrained strictly between 0.0 and 1.0"
    )


class DiagnosticReport(BaseModel):
    observation: str = Field(description="Direct observations synthesized from upstream ML evidence, pump telemetry, and past cases.")
    interpretation: str = Field(description="Suspected mechanical failure mode (e.g., seal degradation, valve leakage, bearing friction, normal operation).")
    recommendation: str = Field(description="Specific physical maintenance or inspection checklist.")
    confidence_assessment: str = Field(description="Evidence-based assessment strictly reflecting the upstream ML confidence without alteration.")
    next_action: str = Field(description="Proposed next workflow action compliant with canonical state machine.")


class LLMDecision(BaseModel):
    action: ActionType
    proposed_state: PumpState
    needs_human_review: bool
    explanation: str = Field(description="1-2 sentences explaining the reasoning.")
    audit_notes: str = Field(description="Internal notes comparing ML vs Context.")
    diagnostic_report: Optional[DiagnosticReport] = Field(
        default=None,
        description="Structured diagnostic report."
    )


class RepairRequest(BaseModel):
    pump_id: str = Field(description="Unique pump identifier to initiate or complete repair.")
    notes: Optional[str] = Field(default=None, description="Technician repair notes or component replacement log.")
    technician_id: Optional[str] = Field(default=None, description="Optional technician identifier.")


class RepairResponse(BaseModel):
    pump_id: str
    previous_state: PumpState
    current_state: PumpState
    action: str
    case_id: str
    message: str
    timestamp: str


class FinalAgentResponse(BaseModel):
    case_id: str
    pump_id: str
    action: ActionType
    current_state: PumpState
    proposed_state: PumpState
    final_state: Optional[PumpState] = None
    ml_prediction: str
    ml_confidence: float
    confidence_flag: Literal["HIGH", "MEDIUM", "LOW"]
    needs_human_review: bool
    explanation: str
    audit_notes: str
    diagnostic_report: Optional[DiagnosticReport] = Field(
        default=None,
        description="Structured diagnostic report."
    )
    was_overridden: bool = Field(
        default=False,
        description="Indicates whether the deterministic state machine overrode an illegal agent proposal."
    )

    def model_post_init(self, __context: Any) -> None:
        if self.final_state is None:
            self.final_state = self.proposed_state


class PumpResponse(BaseModel):
    pump_id: str
    status: PumpState
    location_info: Optional[str] = None
    age_years: Optional[int] = None
    last_maintenance_date: Optional[str] = None
    previous_failures: Optional[int] = None
    known_issues: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None