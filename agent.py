import os
import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from schemas import MLPayload, LLMDecision, FinalAgentResponse, DiagnosticReport
from tools import execute_tool_safely, PUMP_CONTEXT_TOOL, PUMP_HISTORY_TOOL


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EmeraldAgent")

load_dotenv()

# Environment Configurations
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

HIGH_THRESH = float(os.getenv("ML_HIGH_CONFIDENCE_THRESHOLD", "0.70"))
LOW_THRESH = float(os.getenv("ML_LOW_CONFIDENCE_THRESHOLD", "0.40"))

# Initialize OpenAI Client flexibly
client_args: Dict[str, Any] = {"api_key": API_KEY}
if BASE_URL and BASE_URL.strip():
    client_args["base_url"] = BASE_URL.strip()

client = OpenAI(**client_args)

# Deterministic State Machine Transition Matrix
VALID_TRANSITIONS: Dict[str, List[str]] = {
    "HEALTHY": ["DIAGNOSIS_PENDING"],
    "DIAGNOSIS_PENDING": ["MAINTENANCE_REQUIRED", "ESCALATED", "HEALTHY"],
    "MAINTENANCE_REQUIRED": ["REPAIR_IN_PROGRESS", "ESCALATED"],
    "REPAIR_IN_PROGRESS": ["VERIFICATION_PENDING", "ESCALATED"],
    "VERIFICATION_PENDING": ["HEALTHY", "MAINTENANCE_REQUIRED", "ESCALATED"],
    "ESCALATED": ["HEALTHY", "DIAGNOSIS_PENDING"]
}

def get_confidence_flag(score: float) -> str:
    """Deterministically maps numerical ML score to categorical flag."""
    if score >= HIGH_THRESH:
        return "HIGH"
    elif score <= LOW_THRESH:
        return "LOW"
    return "MEDIUM"

def validate_transition(current_state: str, proposed_state: str) -> bool:
    """Deterministically validates whether a state transition is legal."""
    allowed = VALID_TRANSITIONS.get(current_state, [])
    return proposed_state in allowed

def evaluate_case(payload: MLPayload) -> FinalAgentResponse:
    logger.info(f"Processing case {payload.case_id} for pump {payload.pump_id}")
    
    # 1. Deterministic Confidence Flag Calculation
    conf_flag = get_confidence_flag(payload.ml_confidence)
    logger.info(f"ML Confidence Score: {payload.ml_confidence} -> Flag: {conf_flag}")
    
    # 2. System Prompt Definition
    system_prompt = f"""You are the Emerald Boys Maintenance AI Agent.
Your role is to interpret ML anomaly reports for rural handpumps and recommend the next workflow action.

WORKFLOW:
1. Review the ML payload (Prediction: {payload.ml_prediction}, Confidence Flag: {conf_flag}, Score: {payload.ml_confidence:.2f}).
2. MUST call get_pump_context with pump_id='{payload.pump_id}' to fetch pump specs/hardware context.
   You may also call get_pump_history with pump_id='{payload.pump_id}' to retrieve historical cases, state transitions, and previous anomalies.
3. Combine ML payload and retrieved context to propose the next logical maintenance state.
4. Provide structured diagnostic reasoning:
   - observation: What ML telemetry and context evidence was provided.
   - interpretation: The likely mechanical failure mode (e.g. seal degradation, valve leak, normal rhythm).
   - recommendation: Practical maintenance inspection tasks.
   - confidence_assessment: Faithful interpretation of the upstream confidence without modifying the score.
   - next_action: State-machine-safe proposed action.

RULES:
- Never invent missing pump context or ML attributes.
- If context indicates errors/unknown pump, choose action "ESCALATE", proposed_state "ESCALATED", needs_human_review = True.
- Low-confidence predictions (LOW) should generally be ESCALATED.
- High-confidence ABNORMAL results usually propose MAINTENANCE_REQUIRED if current state allows.
- You only PROPOSE state transitions. Python/backend logic enforces transition rules strictly."""

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"ML Payload: {payload.model_dump_json()}"}
    ]

    context_retrieved = False
    diagnostic_report: Optional[DiagnosticReport] = None
    
    try:
        # First turn: Send request and allow tool calling
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[PUMP_CONTEXT_TOOL, PUMP_HISTORY_TOOL]
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # Execute tool calls securely if requested
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments or "{}")
                logger.info(f"Tool call requested: {func_name} with args {func_args}")
                
                tool_output = execute_tool_safely(func_name, func_args)
                if func_name == "get_pump_context":
                    context_retrieved = True
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output
                })

        # Fallback check: If the LLM skipped calling get_pump_context
        if not context_retrieved:
            logger.warning("LLM skipped required tool call. Fallback fetching pump context.")
            tool_output = execute_tool_safely("get_pump_context", {"pump_id": payload.pump_id})
            messages.append({
                "role": "system",
                "content": f"Automated context injection (LLM omitted tool call): {tool_output}"
            })

        # Second turn: Force Structured Pydantic Output (LLMDecision)
        final_completion = client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            response_format=LLMDecision
        )
        
        llm_decision: LLMDecision = final_completion.choices[0].message.parsed
        diagnostic_report = llm_decision.diagnostic_report

    except Exception as e:
        logger.error(f"API/Execution Error during evaluation: {str(e)}")
        fallback_report = DiagnosticReport(
            observation=f"Evaluation failed due to runtime error: {str(e)}",
            interpretation="System error prevented complete automated acoustic reasoning.",
            recommendation="Escalate for technician physical inspection and investigate agent runtime.",
            confidence_assessment=f"Upstream ML Confidence was {payload.ml_confidence:.2f} ({conf_flag}), but agent execution failed.",
            next_action="ESCALATE"
        )
        return FinalAgentResponse(
            case_id=payload.case_id,
            pump_id=payload.pump_id,
            action="ESCALATE",
            current_state=payload.current_state,
            proposed_state="ESCALATED",
            ml_prediction=payload.ml_prediction,
            ml_confidence=payload.ml_confidence,
            confidence_flag=conf_flag,
            needs_human_review=True,
            explanation="System error encountered during agent decision evaluation.",
            audit_notes=f"System Error: {str(e)}",
            diagnostic_report=fallback_report
        )

    # 3. Deterministic Validation & Semantic Safety Logic
    action = llm_decision.action
    proposed_state = llm_decision.proposed_state
    needs_human_review = llm_decision.needs_human_review
    audit_notes = llm_decision.audit_notes
    explanation = llm_decision.explanation

    # Semantic enforcement: ESCALATE action consistency
    if action == "ESCALATE":
        proposed_state = "ESCALATED"
        needs_human_review = True

    # Deterministic State Machine Validation
    is_valid_transition = validate_transition(payload.current_state, proposed_state)

    if not is_valid_transition:
        logger.warning(
            f"REJECTED: Illegal transition proposed ({payload.current_state} -> {proposed_state}). Overriding to ESCALATED."
        )
        action = "ESCALATE"
        proposed_state = "ESCALATED"
        needs_human_review = True
        audit_notes += f" [OVERRIDE]: Proposed illegal transition '{payload.current_state}' -> '{llm_decision.proposed_state}' was rejected by deterministic state machine."

    logger.info(f"Final Decision: Action={action}, Proposed State={proposed_state}, Human Review={needs_human_review}")

    # Ensure structured diagnostic report exists and is consistent with final validated action
    if diagnostic_report is None:
        diagnostic_report = DiagnosticReport(
            observation=f"ML Prediction: {payload.ml_prediction} ({payload.ml_confidence:.2f}). State: {payload.current_state}.",
            interpretation=explanation or "Anomaly evaluation synthesized from ML telemetry.",
            recommendation="Inspect physical handpump mechanism according to maintenance protocol.",
            confidence_assessment=f"Categorized as {conf_flag} confidence based on deterministic thresholding.",
            next_action=action
        )
    elif not is_valid_transition:
        # Reflect backend override in diagnostic report next action
        diagnostic_report.next_action = "ESCALATE (Backend Deterministic Override)"

    # 4. Construct Final Response using TRUSTED UPSTREAM PAYLOAD
    return FinalAgentResponse(
        case_id=payload.case_id,
        pump_id=payload.pump_id,
        action=action,
        current_state=payload.current_state,
        proposed_state=proposed_state,
        ml_prediction=payload.ml_prediction,
        ml_confidence=payload.ml_confidence,
        confidence_flag=conf_flag,
        needs_human_review=needs_human_review,
        explanation=explanation,
        audit_notes=audit_notes,
        diagnostic_report=diagnostic_report
    )