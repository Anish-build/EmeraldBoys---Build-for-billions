import unittest
from pydantic import ValidationError

from schemas import MLPayload
from tools import execute_tool_safely
from agent import get_confidence_flag, validate_transition, evaluate_case

class TestEmeraldAgentV1(unittest.TestCase):

    def test_01_confidence_bounds_high_invalid(self):
        """Verify invalid confidence > 1 raises ValidationError"""
        with self.assertRaises(ValidationError):
            MLPayload(
                case_id="TEST-01",
                pump_id="PUMP-001",
                current_state="DIAGNOSIS_PENDING",
                ml_prediction="ABNORMAL",
                ml_confidence=1.01
            )

    def test_02_confidence_bounds_low_invalid(self):
        """Verify invalid confidence < 0 raises ValidationError"""
        with self.assertRaises(ValidationError):
            MLPayload(
                case_id="TEST-02",
                pump_id="PUMP-001",
                current_state="DIAGNOSIS_PENDING",
                ml_prediction="ABNORMAL",
                ml_confidence=-0.01
            )

    def test_03_confidence_flag_thresholds(self):
        """Verify deterministic get_confidence_flag threshold categorization"""
        self.assertEqual(get_confidence_flag(0.92), "HIGH")
        self.assertEqual(get_confidence_flag(0.70), "HIGH")
        self.assertEqual(get_confidence_flag(0.69), "MEDIUM")
        self.assertEqual(get_confidence_flag(0.40), "LOW")
        self.assertEqual(get_confidence_flag(0.35), "LOW")

    def test_04_state_machine_transitions(self):
        """Verify canonical state transition rules"""
        # Legal transitions
        self.assertTrue(validate_transition("HEALTHY", "DIAGNOSIS_PENDING"))
        self.assertTrue(validate_transition("DIAGNOSIS_PENDING", "MAINTENANCE_REQUIRED"))
        self.assertTrue(validate_transition("DIAGNOSIS_PENDING", "ESCALATED"))
        self.assertTrue(validate_transition("DIAGNOSIS_PENDING", "HEALTHY"))
        
        # Illegal transitions
        self.assertFalse(validate_transition("HEALTHY", "REPAIR_IN_PROGRESS"))
        self.assertFalse(validate_transition("HEALTHY", "MAINTENANCE_REQUIRED"))
        self.assertFalse(validate_transition("MAINTENANCE_REQUIRED", "HEALTHY"))

    def test_05_tools_safety_and_unknown_pump(self):
        """Verify tool execution safety and unknown pump handling"""
        # Known pump
        res = execute_tool_safely("get_pump_context", {"pump_id": "PUMP-001"})
        self.assertIn("age_years", res)

        # Unknown pump
        res_unk = execute_tool_safely("get_pump_context", {"pump_id": "PUMP-999"})
        self.assertIn("Pump not found in database.", res_unk)

        # Unauthorized tool execution attempt
        res_unauth = execute_tool_safely("unauthorized_tool", {})
        self.assertIn("Unauthorized tool call attempted", res_unauth)

    def test_06_illegal_transition_override_in_agent(self):
        """Verify illegal state transitions are rejected and overridden to ESCALATED"""
        # Pump is HEALTHY, but payload is ABNORMAL. HEALTHY can ONLY transition to DIAGNOSIS_PENDING.
        payload = MLPayload(
            case_id="TEST-ILLEGAL",
            pump_id="PUMP-001",
            current_state="HEALTHY",
            ml_prediction="ABNORMAL",
            ml_confidence=0.92
        )
        res = evaluate_case(payload)
        
        # If the model tried to jump straight to MAINTENANCE_REQUIRED, backend must override to ESCALATED
        if res.proposed_state != "DIAGNOSIS_PENDING":
            self.assertEqual(res.proposed_state, "ESCALATED")
            self.assertEqual(res.action, "ESCALATE")
            self.assertTrue(res.needs_human_review)

if __name__ == "__main__":
    unittest.main()