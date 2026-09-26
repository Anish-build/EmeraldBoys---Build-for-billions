import os
import uuid
import unittest
from fastapi.testclient import TestClient
from database import init_db, SessionLocal
import repositories
from schemas import MLPayload, DiagnosticReport, FinalAgentResponse
from agent import evaluate_case
from api import app

class TestEmeraldV4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_01_structured_diagnostic_report_schema(self):
        """V4.2: Structured DiagnosticReport model validates all required diagnostic sections."""
        report = DiagnosticReport(
            observation="Acoustic sensor detected 120 Hz fundamental frequency with 0.92 anomaly score.",
            interpretation="Severe seal degradation indicated by persistent high-frequency chatter.",
            recommendation="Dispatch maintenance technician to inspect and replace pump cylinder seal.",
            confidence_assessment="Upstream ML confidence is 0.92 (HIGH), indicating strong acoustic anomaly.",
            next_action="PROPOSE_TRANSITION"
        )
        self.assertEqual(report.next_action, "PROPOSE_TRANSITION")
        self.assertIn("seal", report.interpretation)

        # Ensure FinalAgentResponse serializes report additively without regression
        res = FinalAgentResponse(
            case_id="TEST-V4-01",
            pump_id="PUMP-001",
            action="PROPOSE_TRANSITION",
            current_state="DIAGNOSIS_PENDING",
            proposed_state="MAINTENANCE_REQUIRED",
            ml_prediction="ABNORMAL",
            ml_confidence=0.92,
            confidence_flag="HIGH",
            needs_human_review=False,
            explanation="Seal degradation detected.",
            audit_notes="Validated by state machine.",
            diagnostic_report=report
        )
        self.assertIsNotNone(res.diagnostic_report)
        self.assertEqual(res.diagnostic_report.observation, report.observation)

    def test_02_agent_produces_structured_diagnostic_report(self):
        """V4.2: evaluate_case produces structured diagnostic report and preserves upstream evidence."""
        payload = MLPayload(
            case_id=f"CASE-V4-{uuid.uuid4().hex[:6]}",
            pump_id="PUMP-001",
            current_state="DIAGNOSIS_PENDING",
            ml_prediction="ABNORMAL",
            ml_confidence=0.92
        )
        response = evaluate_case(payload)
        self.assertIsNotNone(response.diagnostic_report)
        report = response.diagnostic_report
        self.assertTrue(len(report.observation) > 0)
        self.assertTrue(len(report.interpretation) > 0)
        self.assertTrue(len(report.recommendation) > 0)
        self.assertTrue(len(report.confidence_assessment) > 0)
        self.assertTrue(len(report.next_action) > 0)

        # Upstream ML values are never altered
        self.assertEqual(response.ml_prediction, "ABNORMAL")
        self.assertEqual(response.ml_confidence, 0.92)

    def test_03_agent_cannot_alter_upstream_ml_confidence(self):
        """V4.2: Upstream ML confidence cannot be overwritten by agent or override."""
        payload = MLPayload(
            case_id=f"CASE-CONF-{uuid.uuid4().hex[:6]}",
            pump_id="PUMP-001",
            current_state="HEALTHY",  # Illegal transition trigger
            ml_prediction="ABNORMAL",
            ml_confidence=0.95
        )
        response = evaluate_case(payload)
        # Even with override, upstream ML confidence and prediction remain immutable
        self.assertEqual(response.ml_confidence, 0.95)
        self.assertEqual(response.ml_prediction, "ABNORMAL")
        self.assertEqual(response.proposed_state, "ESCALATED")
        self.assertTrue(response.needs_human_review)

    def test_04_repair_start_valid_transition(self):
        """V4.3: POST /repair/start initiates repair from MAINTENANCE_REQUIRED -> REPAIR_IN_PROGRESS."""
        pump_id = f"PUMP-REPAIR-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="MAINTENANCE_REQUIRED")

        res = self.client.post("/repair/start", json={
            "pump_id": pump_id,
            "notes": "Technician Alice dispatched with replacement seals.",
            "technician_id": "TECH-042"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["pump_id"], pump_id)
        self.assertEqual(data["previous_state"], "MAINTENANCE_REQUIRED")
        self.assertEqual(data["current_state"], "REPAIR_IN_PROGRESS")
        self.assertEqual(data["action"], "START_REPAIR")

        # Verify DB status updated and transition recorded
        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "REPAIR_IN_PROGRESS")
        transitions = repositories.get_transitions_for_pump(self.db, pump_id)
        self.assertGreaterEqual(len(transitions), 1)
        self.assertEqual(transitions[0].final_state, "REPAIR_IN_PROGRESS")

    def test_05_repair_start_illegal_state_rejected(self):
        """V4.3: POST /repair/start rejects pumps not in MAINTENANCE_REQUIRED status."""
        pump_id = f"PUMP-HEALTHY-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="HEALTHY")

        res = self.client.post("/repair/start", json={"pump_id": pump_id})
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertIn("Illegal transition request", data["error"])

        # Pump status must not have changed
        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "HEALTHY")

    def test_06_repair_complete_valid_transition(self):
        """V4.3: POST /repair/complete marks repair complete: REPAIR_IN_PROGRESS -> VERIFICATION_PENDING."""
        pump_id = f"PUMP-COMPL-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="REPAIR_IN_PROGRESS")

        res = self.client.post("/repair/complete", json={
            "pump_id": pump_id,
            "notes": "Cylinder rod and cup seals replaced and lubricated."
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["previous_state"], "REPAIR_IN_PROGRESS")
        self.assertEqual(data["current_state"], "VERIFICATION_PENDING")
        self.assertEqual(data["action"], "COMPLETE_REPAIR")

        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "VERIFICATION_PENDING")

    def test_07_repair_complete_illegal_state_rejected(self):
        """V4.3: POST /repair/complete rejects pumps not in REPAIR_IN_PROGRESS status."""
        pump_id = f"PUMP-ILL-COMPL-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="MAINTENANCE_REQUIRED")

        res = self.client.post("/repair/complete", json={"pump_id": pump_id})
        self.assertEqual(res.status_code, 400)

        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "MAINTENANCE_REQUIRED")

    def test_08_verify_endpoint_successful_repair_recovery(self):
        """V4.4: POST /verify confirms successful repair: VERIFICATION_PENDING -> HEALTHY."""
        pump_id = f"PUMP-VER-SUCC-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="VERIFICATION_PENDING")

        case_id = f"CASE-VER-{uuid.uuid4().hex[:6]}"
        verification_payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "VERIFICATION_PENDING",
            "ml_prediction": "NORMAL",
            "ml_confidence": 0.88
        }

        res = self.client.post("/verify", json=verification_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["proposed_state"], "HEALTHY")
        self.assertFalse(data["needs_human_review"])

        # Check DB pump updated to HEALTHY
        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "HEALTHY")

        # Verify DiagnosticCase and StateTransition recorded in SQLite
        saved_case = repositories.get_case_by_id(self.db, case_id)
        self.assertIsNotNone(saved_case)
        self.assertEqual(saved_case.final_state, "HEALTHY")

    def test_09_verify_endpoint_unsuccessful_repair_persistent_fault(self):
        """V4.4: POST /verify detects persistent fault: VERIFICATION_PENDING -> MAINTENANCE_REQUIRED."""
        pump_id = f"PUMP-VER-FAIL-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="VERIFICATION_PENDING")

        case_id = f"CASE-VER-FAIL-{uuid.uuid4().hex[:6]}"
        verification_payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "VERIFICATION_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.85
        }

        res = self.client.post("/verify", json=verification_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["proposed_state"], "MAINTENANCE_REQUIRED")
        self.assertTrue(data["needs_human_review"])

        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "MAINTENANCE_REQUIRED")

    def test_10_verify_endpoint_uncertainty_escalates(self):
        """V4.4: POST /verify with uncertain/low-confidence audio triggers ESCALATED."""
        pump_id = f"PUMP-VER-LOW-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="VERIFICATION_PENDING")

        case_id = f"CASE-VER-UNCERTAIN-{uuid.uuid4().hex[:6]}"
        verification_payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "VERIFICATION_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.32  # LOW confidence trigger
        }

        res = self.client.post("/verify", json=verification_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["proposed_state"], "ESCALATED")
        self.assertTrue(data["needs_human_review"])

        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "ESCALATED")

        from events import get_events_for_pump
        events = get_events_for_pump(pump_id)
        self.assertGreaterEqual(len(events), 1)

    def test_11_verify_endpoint_illegal_when_not_verification_pending(self):
        """V4.4: POST /verify rejects verification attempts on pumps not in VERIFICATION_PENDING."""
        pump_id = f"PUMP-NOT-PEND-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="HEALTHY")

        res = self.client.post("/verify", json={
            "case_id": "CASE-INVALID-VER",
            "pump_id": pump_id,
            "current_state": "VERIFICATION_PENDING",
            "ml_prediction": "NORMAL",
            "ml_confidence": 0.88
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("Illegal verification state", res.json()["error"])

    def test_12_full_lifecycle_history_audit_trail(self):
        """V4.4: Full before/after diagnostic cases and repair transitions preserved in SQLite history."""
        pump_id = "PUMP-001"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="DIAGNOSIS_PENDING")
        repositories.update_pump_status(self.db, pump_id, "DIAGNOSIS_PENDING")

        # 1. Initial Diagnosis: Abnormal (High confidence) -> MAINTENANCE_REQUIRED
        case_1_id = f"CASE-INITIAL-{uuid.uuid4().hex[:6]}"
        res1 = self.client.post("/evaluate", json={
            "case_id": case_1_id,
            "pump_id": pump_id,
            "current_state": "DIAGNOSIS_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.92
        })
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["proposed_state"], "MAINTENANCE_REQUIRED")

        # 2. Repair Start -> REPAIR_IN_PROGRESS
        res2 = self.client.post("/repair/start", json={"pump_id": pump_id, "notes": "Dispatched mechanic."})
        self.assertEqual(res2.status_code, 200)

        # 3. Repair Complete -> VERIFICATION_PENDING
        res3 = self.client.post("/repair/complete", json={"pump_id": pump_id, "notes": "Parts replaced."})
        self.assertEqual(res3.status_code, 200)

        # 4. Post-Repair Verification: Normal -> HEALTHY
        case_2_id = f"CASE-VERIFY-{uuid.uuid4().hex[:6]}"
        res4 = self.client.post("/verify", json={
            "case_id": case_2_id,
            "pump_id": pump_id,
            "current_state": "VERIFICATION_PENDING",
            "ml_prediction": "NORMAL",
            "ml_confidence": 0.88
        })
        self.assertEqual(res4.status_code, 200)
        self.assertEqual(res4.json()["proposed_state"], "HEALTHY")

        # 5. Query Pump History via Endpoint
        res_history = self.client.get(f"/pumps/{pump_id}/history")
        self.assertEqual(res_history.status_code, 200)
        history = res_history.json()
        self.assertEqual(history["current_status"], "HEALTHY")
        self.assertGreaterEqual(history["cases_count"], 2)

        # Both cases must be present
        case_ids = [c["case_id"] for c in history["cases"]]
        self.assertIn(case_1_id, case_ids)
        self.assertIn(case_2_id, case_ids)

        # All 4 state transitions must be present
        actions = [t["action"] for t in history["transitions"]]
        self.assertIn("START_REPAIR", actions)
        self.assertIn("COMPLETE_REPAIR", actions)

if __name__ == "__main__":
    unittest.main()
