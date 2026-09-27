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
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AGENT_MODE = os.getenv("AGENT_MODE", "auto").lower()

HIGH_THRESH = float(os.getenv("ML_HIGH_CONFIDENCE_THRESHOLD", "0.70"))
LOW_THRESH = float(os.getenv("ML_LOW_CONFIDENCE_THRESHOLD", "0.40"))

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


def get_openai_client() -> Optional[OpenAI]:
    """
    Safely initializes OpenAI client only if a valid, non-placeholder API key exists.
    Returns None if missing, preventing import crashes and enabling local prototype mode.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        return None
    cleaned_key = api_key.strip()
    if cleaned_key in ("your_key_here", "mock", "test", "none", "YOUR_OPENROUTER_OR_OPENAI_KEY"):
        return None
    
    client_args: Dict[str, Any] = {"api_key": cleaned_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and base_url.strip():
        client_args["base_url"] = base_url.strip()
    
    try:
        return OpenAI(**client_args)
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {e}")
        return None


def evaluate_case(payload: MLPayload) -> FinalAgentResponse:
    """
    Executes AI Agent reasoning on the upstream MLPayload, retrieves persistent context/history
    via controlled tools, and enforces deterministic state validation.
    
    Safeguards:
    - Never modifies upstream ML confidence or prediction.
    - Survives missing/invalid API keys and network errors with safe escalation fallback.
    - Overrides illegal transition proposals while preserving the raw proposed_state.
    """
    logger.info(f"Processing case {payload.case_id} for pump {payload.pump_id}")
    
    # 1. Deterministic Confidence Flag Calculation
    conf_flag = get_confidence_flag(payload.ml_confidence)
    logger.info(f"ML Confidence Score: {payload.ml_confidence} -> Flag: {conf_flag}")
    
    client = None
    if AGENT_MODE != "mock":
        client = get_openai_client()

    diagnostic_report: Optional[DiagnosticReport] = None
    raw_proposed_state: str = "ESCALATED"
    action: str = "ESCALATE"
    needs_human_review: bool = True
    explanation: str = ""
    audit_notes: str = ""

    # Mode A: Live LLM Agent Execution via OpenAI / OpenRouter
    if client is not None:
        logger.info(f"Invoking Live LLM Agent ({MODEL}) with function calling tools.")
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
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[PUMP_CONTEXT_TOOL, PUMP_HISTORY_TOOL]
            )
            msg = response.choices[0].message
            messages.append(msg)
            
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

            if not context_retrieved:
                tool_output = execute_tool_safely("get_pump_context", {"pump_id": payload.pump_id})
                messages.append({
                    "role": "system",
                    "content": f"Automated context injection: {tool_output}"
                })

            final_completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=messages,
                response_format=LLMDecision
            )
            llm_decision: LLMDecision = final_completion.choices[0].message.parsed
            action = llm_decision.action
            raw_proposed_state = llm_decision.proposed_state
            needs_human_review = llm_decision.needs_human_review
            explanation = llm_decision.explanation
            audit_notes = llm_decision.audit_notes
            diagnostic_report = llm_decision.diagnostic_report

        except Exception as e:
            logger.error(f"Live LLM Agent failed safely: {str(e)}")
            fallback_report = DiagnosticReport(
                observation=f"Evaluation failed due to runtime error: {str(e)}",
                interpretation="System error prevented complete automated acoustic reasoning.",
                recommendation="Escalate for technician physical inspection and investigate agent runtime.",
                confidence_assessment=f"Upstream ML Confidence was {payload.ml_confidence:.2f} ({conf_flag}), but agent execution failed.",
                next_action="ESCALATE"
            )
            raw_proposed_state = "ESCALATED"
            action = "ESCALATE"
            needs_human_review = True
            explanation = "System error encountered during agent decision evaluation."
            audit_notes = f"System Error: {str(e)}"
            diagnostic_report = fallback_report

    # Mode B: Transparent Deterministic Prototype Agent (Local / Offline Fallback)
    else:
        logger.info("Executing Transparent Prototype Agent reasoning with database context retrieval.")
        context_str = execute_tool_safely("get_pump_context", {"pump_id": payload.pump_id})
        history_str = execute_tool_safely("get_pump_history", {"pump_id": payload.pump_id})
        
        context_data = {}
        try:
            context_data = json.loads(context_str)
        except Exception:
            pass

        # Case: Pump context lookup failed / unknown pump
        if "error" in context_data:
            action = "ESCALATE"
            raw_proposed_state = "ESCALATED"
            needs_human_review = True
            explanation = f"Pump context fetch failed for {payload.pump_id}: {context_data.get('error')}"
            audit_notes = f"Unknown pump ID '{payload.pump_id}'. Mandatory context tool returned error: {context_data.get('error')}"
            diagnostic_report = DiagnosticReport(
                observation=f"ML Prediction: {payload.ml_prediction} ({payload.ml_confidence:.2f}). Context fetch failed for {payload.pump_id}.",
                interpretation="Unregistered hardware identifier; acoustic baseline cannot be safely correlated.",
                recommendation="Verify physical pump serial number and register hardware profile in database.",
                confidence_assessment=f"ML Confidence is {payload.ml_confidence:.2f} ({conf_flag}), but hardware context is missing.",
                next_action="ESCALATE"
            )

        # Case: Low-confidence prediction (uncertainty escalation)
        elif conf_flag == "LOW":
            action = "ESCALATE"
            raw_proposed_state = "ESCALATED"
            needs_human_review = True
            explanation = f"Low confidence acoustic score ({payload.ml_confidence:.2f}) indicates significant diagnostic ambiguity."
            audit_notes = f"Categorized as LOW confidence ({payload.ml_confidence:.2f} <= {LOW_THRESH}). Automatic escalation triggered."
            diagnostic_report = DiagnosticReport(
                observation=f"Acoustic anomaly detected but probability score is LOW ({payload.ml_confidence:.2f}).",
                interpretation="Inconclusive acoustic telemetry; possible intermittent mechanical chatter or background environmental noise.",
                recommendation="Dispatch field technician for manual acoustic listening and physical pump handle stroke testing.",
                confidence_assessment=f"Low confidence ({payload.ml_confidence:.2f}) mandates manual human review.",
                next_action="ESCALATE"
            )

        # Case: High/Medium confidence ABNORMAL prediction
        elif payload.ml_prediction == "ABNORMAL":
            action = "PROPOSE_TRANSITION"
            raw_proposed_state = "MAINTENANCE_REQUIRED"
            needs_human_review = False
            known = context_data.get("known_issues", "Component wear")
            explanation = f"Acoustic anomaly confirmed with {payload.ml_confidence:.2f} confidence. Hardware history notes: {known}."
            audit_notes = f"High-confidence anomaly ({payload.ml_confidence:.2f}). Proposing transition to MAINTENANCE_REQUIRED."
            diagnostic_report = DiagnosticReport(
                observation=f"Acoustic sensor detected anomaly with {payload.ml_confidence:.2f} probability ({conf_flag} confidence).",
                interpretation=f"Suspected mechanical failure: {known} or cylinder seal friction.",
                recommendation="Inspect pump cylinder, replace worn piston leather seals, and check rod alignment.",
                confidence_assessment=f"High acoustic confidence ({payload.ml_confidence:.2f}) indicates clear mechanical defect.",
                next_action="PROPOSE_TRANSITION"
            )

        # Case: High/Medium confidence NORMAL prediction
        else:
            action = "PROPOSE_TRANSITION"
            raw_proposed_state = "HEALTHY"
            needs_human_review = False
            explanation = f"Acoustic signature confirms NORMAL handpump operation with {payload.ml_confidence:.2f} confidence."
            audit_notes = f"Normal acoustic rhythm detected with {payload.ml_confidence:.2f} confidence. Proposing transition to HEALTHY."
            diagnostic_report = DiagnosticReport(
                observation=f"Periodic acoustic rhythm within normal baseline tolerances ({payload.ml_confidence:.2f} confidence).",
                interpretation="Standard operational acoustic profile; no mechanical friction or leakage detected.",
                recommendation="Record regular routine inspection log; no immediate maintenance required.",
                confidence_assessment=f"High acoustic confidence ({payload.ml_confidence:.2f}) confirms normal operation.",
                next_action="PROPOSE_TRANSITION"
            )

    # 3. Deterministic Validation & Semantic Safety Logic
    if action == "ESCALATE":
        raw_proposed_state = "ESCALATED"
        needs_human_review = True

    # Deterministic State Machine Validation
    is_valid_transition = validate_transition(payload.current_state, raw_proposed_state)

    if not is_valid_transition:
        logger.warning(
            f"REJECTED: Illegal transition proposed ({payload.current_state} -> {raw_proposed_state}). Overriding final_state to ESCALATED."
        )
        was_overridden = True
        final_state = "ESCALATED"
        action = "ESCALATE"
        needs_human_review = True
        audit_notes += f" [OVERRIDE]: Proposed illegal transition '{payload.current_state}' -> '{raw_proposed_state}' was rejected by deterministic state machine."
        if diagnostic_report:
            diagnostic_report.next_action = "ESCALATE (Backend Deterministic Override)"
    else:
        was_overridden = False
        final_state = raw_proposed_state

    logger.info(
        f"Final Decision for {payload.case_id}: Action={action}, Proposed={raw_proposed_state}, Final={final_state}, Overridden={was_overridden}"
    )

    if diagnostic_report is None:
        diagnostic_report = DiagnosticReport(
            observation=f"ML Prediction: {payload.ml_prediction} ({payload.ml_confidence:.2f}). State: {payload.current_state}.",
            interpretation=explanation or "Anomaly evaluation synthesized from ML telemetry.",
            recommendation="Inspect physical handpump mechanism according to maintenance protocol.",
            confidence_assessment=f"Categorized as {conf_flag} confidence based on deterministic thresholding.",
            next_action=action
        )

    # 4. Construct Final Response preserving both raw proposed_state and validated final_state
    return FinalAgentResponse(
        case_id=payload.case_id,
        pump_id=payload.pump_id,
        action=action,
        current_state=payload.current_state,
        proposed_state=raw_proposed_state,
        final_state=final_state,
        ml_prediction=payload.ml_prediction,
        ml_confidence=payload.ml_confidence,
        confidence_flag=conf_flag,
        needs_human_review=needs_human_review,
        explanation=explanation,
        audit_notes=audit_notes,
        diagnostic_report=diagnostic_report,
        was_overridden=was_overridden
    )