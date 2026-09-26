from pydantic import BaseModel, Field
from typing import Literal

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
    case_id: str
    pump_id: str
    current_state: PumpState
    ml_prediction: str
    ml_confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="ML model probability score constrained between 0.0 and 1.0"
    )

class LLMDecision(BaseModel):
    action: ActionType
    proposed_state: PumpState
    needs_human_review: bool
    explanation: str = Field(description="1-2 sentences explaining the reasoning.")
    audit_notes: str = Field(description="Internal notes comparing ML vs Context.")

class FinalAgentResponse(BaseModel):
    case_id: str
    pump_id: str
    action: ActionType
    current_state: PumpState
    proposed_state: PumpState
    ml_prediction: str
    ml_confidence: float
    confidence_flag: Literal["HIGH", "MEDIUM", "LOW"]
    needs_human_review: bool
    explanation: str
    audit_notes: str