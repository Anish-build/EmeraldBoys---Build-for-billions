import os
import unittest
from fastapi.testclient import TestClient
from database import init_db, SessionLocal, DATABASE_PATH, Pump, DiagnosticCase, StateTransition
import repositories
from api import app

class TestEmeraldV3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_01_database_file_created(self):
        """Verify that data/emerald_boys.db exists on disk."""
        self.assertTrue(os.path.exists(DATABASE_PATH), "Database file does not exist at expected path.")

    def test_02_pump_creation_and_retrieval(self):
        """Verify creating and retrieving pump records."""
        test_pump_id = "PUMP-TEST-001"
        pump = repositories.get_or_create_pump(self.db, pump_id=test_pump_id, status="HEALTHY", location_info="Sector 4")
        self.assertIsNotNone(pump)
        self.assertEqual(pump.pump_id, test_pump_id)
        self.assertEqual(pump.status, "HEALTHY")

        # Retrieving again should fetch the existing pump without duplicating
        pump_fetched = repositories.get_or_create_pump(self.db, pump_id=test_pump_id)
        self.assertEqual(pump_fetched.id, pump.id)

    def test_03_save_diagnostic_case(self):
        """Verify saving and retrieving a diagnostic case."""
        test_pump_id = "PUMP-TEST-002"
        repositories.get_or_create_pump(self.db, pump_id=test_pump_id, status="DIAGNOSIS_PENDING")

        diag_case = repositories.save_diagnostic_case(
            db=self.db,
            case_id="CASE-V3-101",
            pump_id=test_pump_id,
            ml_prediction="ABNORMAL",
            ml_confidence=0.92,
            confidence_flag="HIGH",
            input_state="DIAGNOSIS_PENDING",
            proposed_state="MAINTENANCE_REQUIRED",
            final_state="MAINTENANCE_REQUIRED",
            action="PROPOSE_TRANSITION",
            needs_human_review=False,
            explanation="High anomaly confidence indicates seal failure.",
            audit_notes="Validated by state machine."
        )

        self.assertIsNotNone(diag_case.id)
        fetched = repositories.get_case_by_id(self.db, "CASE-V3-101")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.case_id, "CASE-V3-101")
        self.assertEqual(fetched.final_state, "MAINTENANCE_REQUIRED")
        self.assertEqual(fetched.confidence_flag, "HIGH")

    def test_04_save_state_transition(self):
        """Verify recording and querying state transitions."""
        test_pump_id = "PUMP-TEST-003"
        repositories.get_or_create_pump(self.db, pump_id=test_pump_id, status="HEALTHY")

        transition = repositories.record_state_transition(
            db=self.db,
            case_id="CASE-V3-102",
            pump_id=test_pump_id,
            from_state="HEALTHY",
            proposed_state="MAINTENANCE_REQUIRED",
            final_state="ESCALATED",
            action="ESCALATE",
            reason="Illegal transition override",
            was_overridden=True
        )

        self.assertIsNotNone(transition.id)
        self.assertTrue(transition.was_overridden)
        self.assertEqual(transition.final_state, "ESCALATED")

        pump_transitions = repositories.get_transitions_for_pump(self.db, test_pump_id)
        self.assertGreaterEqual(len(pump_transitions), 1)
        self.assertEqual(pump_transitions[0].case_id, "CASE-V3-102")

    def test_05_multiple_cases_preserved_no_overwrite(self):
        """Data integrity test: Multiple cases for same pump must both persist without overwrite."""
        pump_id = "PUMP-MULTI-TEST"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="DIAGNOSIS_PENDING")

        # Case 1
        case1 = repositories.save_diagnostic_case(
            db=self.db,
            case_id="CASE-MULTI-01",
            pump_id=pump_id,
            ml_prediction="ABNORMAL",
            ml_confidence=0.92,
            confidence_flag="HIGH",
            input_state="DIAGNOSIS_PENDING",
            proposed_state="MAINTENANCE_REQUIRED",
            final_state="MAINTENANCE_REQUIRED",
            action="PROPOSE_TRANSITION",
            needs_human_review=False,
            explanation="Initial anomaly evaluation.",
            audit_notes="First run."
        )

        # Case 2
        case2 = repositories.save_diagnostic_case(
            db=self.db,
            case_id="CASE-MULTI-02",
            pump_id=pump_id,
            ml_prediction="NORMAL",
            ml_confidence=0.88,
            confidence_flag="HIGH",
            input_state="DIAGNOSIS_PENDING",
            proposed_state="HEALTHY",
            final_state="HEALTHY",
            action="PROPOSE_TRANSITION",
            needs_human_review=False,
            explanation="Followup verification evaluation.",
            audit_notes="Second run."
        )

        # Both cases must exist
        all_cases = repositories.get_cases_for_pump(self.db, pump_id)
        case_ids = [c.case_id for c in all_cases]
        self.assertIn("CASE-MULTI-01", case_ids)
        self.assertIn("CASE-MULTI-02", case_ids)
        self.assertGreaterEqual(len(all_cases), 2)

    def test_06_evaluate_endpoint_persists_case_and_transition(self):
        """Step 2: POST /evaluate persists DiagnosticCase and StateTransition into SQLite."""
        import uuid
        case_id = f"CASE-PERSIST-{uuid.uuid4().hex[:8]}"
        pump_id = "PUMP-001"
        payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "DIAGNOSIS_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.92
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)

        # Query DB directly to verify persistence
        saved_case = repositories.get_case_by_id(self.db, case_id)
        self.assertIsNotNone(saved_case)
        self.assertEqual(saved_case.case_id, case_id)
        self.assertEqual(saved_case.pump_id, pump_id)
        self.assertEqual(saved_case.final_state, "MAINTENANCE_REQUIRED")

        transitions = repositories.get_transitions_for_case(self.db, case_id)
        self.assertGreaterEqual(len(transitions), 1)
        self.assertEqual(transitions[0].from_state, "DIAGNOSIS_PENDING")
        self.assertEqual(transitions[0].final_state, "MAINTENANCE_REQUIRED")
        self.assertFalse(transitions[0].was_overridden)

        # Pump status should match final state
        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertIsNotNone(pump)
        self.assertEqual(pump.status, "MAINTENANCE_REQUIRED")

    def test_07_evaluate_endpoint_persists_illegal_override(self):
        """Step 2: POST /evaluate persists illegal transition override with was_overridden=True."""
        import uuid
        case_id = f"CASE-ILLEGAL-{uuid.uuid4().hex[:8]}"
        pump_id = "PUMP-001"
        payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "HEALTHY",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.95
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)

        saved_case = repositories.get_case_by_id(self.db, case_id)
        self.assertIsNotNone(saved_case)
        self.assertEqual(saved_case.final_state, "ESCALATED")
        self.assertTrue(saved_case.needs_human_review)

        transitions = repositories.get_transitions_for_case(self.db, case_id)
        self.assertGreaterEqual(len(transitions), 1)
        self.assertEqual(transitions[0].from_state, "HEALTHY")
        self.assertEqual(transitions[0].final_state, "ESCALATED")
        self.assertTrue(transitions[0].was_overridden)

    def test_08_get_pump_history_tool(self):
        """Step 3: get_pump_history retrieves previous cases, states, and anomalies from DB."""
        import json
        from tools import execute_tool_safely

        # Query history for PUMP-001 which has persisted cases from test_06/07
        res_raw = execute_tool_safely("get_pump_history", {"pump_id": "PUMP-001"})
        res = json.loads(res_raw)
        self.assertEqual(res.get("pump_id"), "PUMP-001")
        self.assertIn("previous_cases", res)
        self.assertIn("recent_states", res)
        self.assertIn("previous_anomalies", res)
        self.assertGreaterEqual(res["previous_cases"], 1)

    def test_09_pump_history_tool_safety(self):
        """Step 3: Tool safety handles invalid args and unknown tools."""
        import json
        from tools import execute_tool_safely

        # Missing pump_id
        res_missing = json.loads(execute_tool_safely("get_pump_history", {}))
        self.assertIn("error", res_missing)

        # Unauthorized tool
        res_unauth = json.loads(execute_tool_safely("execute_arbitrary_sql", {"query": "DROP TABLE pumps"}))
        self.assertIn("Unauthorized tool call attempted", res_unauth["error"])

    def test_10_ml_adapter_predict(self):
        """Step 3: MLAdapter.predict generates valid MLPayload from audio paths deterministically."""
        from ml_adapter import MLAdapter, predict, ADAPTER_MODE
        from schemas import MLPayload

        self.assertEqual(ADAPTER_MODE, "MOCK ML")
        self.assertEqual(MLAdapter.MODE, "MOCK ML")

        # Normal audio profile
        payload_normal = predict("samples/audio/normal_pump_sample.wav", pump_id="PUMP-001")
        self.assertIsInstance(payload_normal, MLPayload)
        self.assertEqual(payload_normal.ml_prediction, "NORMAL")
        self.assertEqual(payload_normal.ml_confidence, 0.88)
        self.assertEqual(payload_normal.pump_id, "PUMP-001")
        self.assertTrue(0.0 <= payload_normal.ml_confidence <= 1.0)

        # High confidence abnormal profile
        payload_abnormal = predict("seal_leak_anomaly.wav", pump_id="PUMP-002")
        self.assertEqual(payload_abnormal.ml_prediction, "ABNORMAL")
        self.assertEqual(payload_abnormal.ml_confidence, 0.92)

        # Subtle leak / low confidence profile
        payload_subtle = predict("low_confidence_chatter.wav", pump_id="PUMP-003")
        self.assertEqual(payload_subtle.ml_prediction, "ABNORMAL")
        self.assertEqual(payload_subtle.ml_confidence, 0.35)

        # Bearing chatter profile
        payload_bearing = predict("bearing_chatter.wav", pump_id="PUMP-004")
        self.assertEqual(payload_bearing.ml_prediction, "ABNORMAL")
        self.assertEqual(payload_bearing.ml_confidence, 0.78)

    def test_11_ml_adapter_invalid_input(self):
        """Step 3: MLAdapter handles invalid inputs safely."""
        from ml_adapter import predict, run_inference

        with self.assertRaises(ValueError):
            predict("")

        with self.assertRaises(ValueError):
            predict(None)

        with self.assertRaises(ValueError):
            run_inference("not-a-dict")

        with self.assertRaises(ValueError):
            run_inference({})

    def test_12_ml_adapter_run_inference(self):
        """Step 3: run_inference produces valid MLPayload with audio_profile key."""
        from ml_adapter import run_inference
        from schemas import MLPayload

        input_data = {
            "pump_id": "PUMP-001",
            "current_state": "DIAGNOSIS_PENDING",
            "audio_profile": "normal_rhythm"
        }
        payload = run_inference(input_data)
        self.assertIsInstance(payload, MLPayload)
        self.assertEqual(payload.ml_prediction, "NORMAL")
        self.assertEqual(payload.ml_confidence, 0.88)

    def test_13_escalation_event_generation(self):
        """Step 4: record_escalation_event creates event ONLY when final_state is ESCALATED."""
        from events import record_escalation_event, get_escalation_events, clear_events

        clear_events()

        # Non-escalated final state should NOT generate an event
        res_none = record_escalation_event(
            pump_id="PUMP-001",
            case_id="CASE-NORM",
            reason="Pump operating normally.",
            final_state="HEALTHY"
        )
        self.assertIsNone(res_none)
        self.assertEqual(len(get_escalation_events()), 0)

        # Valid escalated final state generates an event
        event = record_escalation_event(
            pump_id="PUMP-001",
            case_id="CASE-ESC-01",
            reason="Severe acoustic anomaly with low confidence.",
            final_state="ESCALATED"
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["pump_id"], "PUMP-001")
        self.assertEqual(event["case_id"], "CASE-ESC-01")
        self.assertEqual(event["final_state"], "ESCALATED")
        self.assertTrue(event["event_id"].startswith("EVT-"))
        self.assertEqual(event["event_type"], "PUMP_ESCALATION")
        self.assertIn("timestamp", event)

        # Verify stored in local store
        events_all = get_escalation_events()
        self.assertEqual(len(events_all), 1)

    def test_14_escalation_webhook_disabled(self):
        """Step 4: When ESCALATION_WEBHOOK_URL is unset, no HTTP request is made."""
        from events import record_escalation_event, clear_events
        clear_events()

        old_url = os.environ.pop("ESCALATION_WEBHOOK_URL", None)
        try:
            event = record_escalation_event(
                pump_id="PUMP-002",
                case_id="CASE-ESC-02",
                reason="Uncertain diagnosis.",
                final_state="ESCALATED"
            )
            self.assertIsNotNone(event)
            self.assertEqual(event["webhook_status"], "DISABLED")
        finally:
            if old_url:
                os.environ["ESCALATION_WEBHOOK_URL"] = old_url

    def test_15_escalation_webhook_safe_handling(self):
        """Step 4: Webhook success and failure do not break workflow."""
        from unittest.mock import patch, MagicMock
        from events import record_escalation_event, clear_events
        import urllib.error

        clear_events()

        # Test Webhook Success
        with patch.dict(os.environ, {"ESCALATION_WEBHOOK_URL": "http://mock-webhook.local/alerts"}):
            mock_resp = MagicMock()
            mock_resp.getcode.return_value = 200
            mock_resp.__enter__.return_value = mock_resp

            with patch("urllib.request.urlopen", return_value=mock_resp):
                event_success = record_escalation_event(
                    pump_id="PUMP-001",
                    case_id="CASE-ESC-SUCC",
                    reason="Escalated case",
                    final_state="ESCALATED"
                )
                self.assertIsNotNone(event_success)
                self.assertEqual(event_success["webhook_status"], "SENT_200")

        # Test Webhook Failure (Network error must not raise)
        with patch.dict(os.environ, {"ESCALATION_WEBHOOK_URL": "http://unreachable-host.local/alerts"}):
            with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
                event_fail = record_escalation_event(
                    pump_id="PUMP-001",
                    case_id="CASE-ESC-FAIL",
                    reason="Escalated case with network issue",
                    final_state="ESCALATED"
                )
                self.assertIsNotNone(event_fail)
                self.assertTrue(event_fail["webhook_status"].startswith("FAILED"))

    def test_16_evaluate_triggers_escalation_event_in_db_and_events(self):
        """Step 4: POST /evaluate triggering ESCALATED records escalation event."""
        import uuid
        from events import get_events_for_pump

        case_id = f"CASE-ESC-EVAL-{uuid.uuid4().hex[:8]}"
        pump_id = "PUMP-ESC-TEST"
        payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "HEALTHY",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.95
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["proposed_state"], "ESCALATED")
        self.assertEqual(data["final_state"], "ESCALATED")

        # Verify escalation event was generated and stored
        pump_events = get_events_for_pump(pump_id)
        self.assertGreaterEqual(len(pump_events), 1)
        matching = [e for e in pump_events if e["case_id"] == case_id]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["final_state"], "ESCALATED")

    def test_17_get_pump_history_api_endpoint(self):
        """Step 5: GET /pumps/{pump_id}/history returns full case and transition history."""
        # Query existing pump PUMP-001
        response = self.client.get("/pumps/PUMP-001/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["pump_id"], "PUMP-001")
        self.assertIn("cases", data)
        self.assertIn("transitions", data)
        self.assertIn("escalation_events", data)
        self.assertGreaterEqual(data["cases_count"], 1)

        # 404 for unknown pump
        resp_404 = self.client.get("/pumps/NON-EXISTENT-PUMP-9999/history")
        self.assertEqual(resp_404.status_code, 404)

    def test_18_repository_update_and_lookup(self):
        """Step 1: Direct repository update_pump_status and get_pump_by_id checks."""
        import uuid
        test_pump_id = f"PUMP-REPO-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=test_pump_id, status="HEALTHY")
        
        pump = repositories.get_pump_by_id(self.db, test_pump_id)
        self.assertIsNotNone(pump)
        self.assertEqual(pump.status, "HEALTHY")

        updated = repositories.update_pump_status(self.db, test_pump_id, "REPAIR_IN_PROGRESS")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "REPAIR_IN_PROGRESS")

        refetched = repositories.get_pump_by_id(self.db, test_pump_id)
        self.assertEqual(refetched.status, "REPAIR_IN_PROGRESS")

if __name__ == "__main__":
    unittest.main()

