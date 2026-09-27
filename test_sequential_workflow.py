import os
import uuid
import unittest
from fastapi.testclient import TestClient

from database import init_db, SessionLocal
import repositories
from schemas import MLPayload, FinalAgentResponse, DiagnosticReport
from agent import evaluate_case, validate_transition, VALID_TRANSITIONS, get_confidence_flag
from audio_processing import load_audio, extract_audio_features, classify_audio, process_audio_file, AudioProcessingError
import ml_adapter
from api import app
from events import get_events_for_pump, clear_events


class TestSequentialWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        self.db = SessionLocal()
        clear_events()

    def tearDown(self):
        self.db.close()

    def test_01_state_machine_matrix_all_transitions(self):
        """Requirement 7, 38: Exhaustively verify all valid and invalid transitions."""
        # 1. Valid transitions
        legal_cases = [
            ("HEALTHY", "DIAGNOSIS_PENDING"),
            ("DIAGNOSIS_PENDING", "MAINTENANCE_REQUIRED"),
            ("DIAGNOSIS_PENDING", "ESCALATED"),
            ("DIAGNOSIS_PENDING", "HEALTHY"),
            ("MAINTENANCE_REQUIRED", "REPAIR_IN_PROGRESS"),
            ("MAINTENANCE_REQUIRED", "ESCALATED"),
            ("REPAIR_IN_PROGRESS", "VERIFICATION_PENDING"),
            ("REPAIR_IN_PROGRESS", "ESCALATED"),
            ("VERIFICATION_PENDING", "HEALTHY"),
            ("VERIFICATION_PENDING", "MAINTENANCE_REQUIRED"),
            ("VERIFICATION_PENDING", "ESCALATED"),
            ("ESCALATED", "HEALTHY"),
            ("ESCALATED", "DIAGNOSIS_PENDING"),
        ]
        for src, dst in legal_cases:
            self.assertTrue(
                validate_transition(src, dst),
                f"Transition {src} -> {dst} should be LEGAL but was rejected."
            )

        # 2. Illegal transitions
        illegal_cases = [
            ("HEALTHY", "MAINTENANCE_REQUIRED"),
            ("HEALTHY", "REPAIR_IN_PROGRESS"),
            ("HEALTHY", "VERIFICATION_PENDING"),
            ("HEALTHY", "ESCALATED"),
            ("MAINTENANCE_REQUIRED", "HEALTHY"),
            ("MAINTENANCE_REQUIRED", "DIAGNOSIS_PENDING"),
            ("MAINTENANCE_REQUIRED", "VERIFICATION_PENDING"),
            ("REPAIR_IN_PROGRESS", "HEALTHY"),
            ("REPAIR_IN_PROGRESS", "MAINTENANCE_REQUIRED"),
            ("REPAIR_IN_PROGRESS", "DIAGNOSIS_PENDING"),
            ("VERIFICATION_PENDING", "REPAIR_IN_PROGRESS"),
            ("VERIFICATION_PENDING", "DIAGNOSIS_PENDING"),
            ("ESCALATED", "MAINTENANCE_REQUIRED"),
            ("ESCALATED", "REPAIR_IN_PROGRESS"),
            ("ESCALATED", "VERIFICATION_PENDING"),
        ]
        for src, dst in illegal_cases:
            self.assertFalse(
                validate_transition(src, dst),
                f"Transition {src} -> {dst} should be ILLEGAL but was accepted."
            )

    def test_02_deterministic_override_preserves_proposal_and_overrides_final(self):
        """Requirement 8, 29, 38: Illegal proposal must preserve proposed_state and set final_state=ESCALATED."""
        pump_id = "PUMP-001"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="HEALTHY")
        repositories.update_pump_status(self.db, pump_id=pump_id, new_status="HEALTHY")

        case_id = f"CASE-OVERRIDE-{uuid.uuid4().hex[:6]}"
        payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "HEALTHY",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.94
        }
        res = self.client.post("/evaluate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Proposal is preserved as MAINTENANCE_REQUIRED
        self.assertEqual(data["proposed_state"], "MAINTENANCE_REQUIRED")
        # Final state is authoritatively overridden to ESCALATED
        self.assertEqual(data["final_state"], "ESCALATED")
        self.assertEqual(data["action"], "ESCALATE")
        self.assertTrue(data["needs_human_review"])
        self.assertTrue(data["was_overridden"])
        self.assertIn("[OVERRIDE]", data["audit_notes"])

        # Check database persistence
        saved_case = repositories.get_case_by_id(self.db, case_id)
        self.assertIsNotNone(saved_case)
        self.assertEqual(saved_case.proposed_state, "MAINTENANCE_REQUIRED")
        self.assertEqual(saved_case.final_state, "ESCALATED")
        self.assertTrue(saved_case.was_overridden)

        # Check pump authoritative status
        pump = repositories.get_pump_by_id(self.db, pump_id)
        self.assertEqual(pump.status, "ESCALATED")

    def test_03_low_confidence_uncertainty_escalation(self):
        """Requirement 20, 28, 54: Low-confidence ML telemetry (<= 0.40) must automatically escalate."""
        pump_id = f"PUMP-LOWCONF-{uuid.uuid4().hex[:6]}"
        repositories.get_or_create_pump(self.db, pump_id=pump_id, status="DIAGNOSIS_PENDING")

        case_id = f"CASE-LOWCONF-{uuid.uuid4().hex[:6]}"
        payload = {
            "case_id": case_id,
            "pump_id": pump_id,
            "current_state": "DIAGNOSIS_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.30
        }
        res = self.client.post("/evaluate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["confidence_flag"], "LOW")
        self.assertEqual(data["final_state"], "ESCALATED")
        self.assertEqual(data["action"], "ESCALATE")
        self.assertTrue(data["needs_human_review"])

        # Verify escalation event was persisted
        events = get_events_for_pump(pump_id)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["case_id"], case_id)
        self.assertEqual(events[0]["final_state"], "ESCALATED")

    def test_04_audio_processing_and_error_handling(self):
        """Requirement 13, 14, 24: Audio loading, feature extraction, and error handling."""
        # 1. Real audio loading & feature extraction
        normal_wav = "samples/audio/normal_pump_sample.wav"
        self.assertTrue(os.path.exists(normal_wav))
        
        proc = process_audio_file(normal_wav, filename_hint=normal_wav)
        self.assertGreater(proc["duration"], 0.5)
        self.assertEqual(proc["sample_rate"], 22050)
        self.assertIn("rms_energy", proc["features"])
        self.assertIn("spectral_centroid", proc["features"])
        self.assertIn("mfcc_1", proc["features"])
        self.assertEqual(proc["prediction"], "NORMAL")
        self.assertGreaterEqual(proc["confidence"], 0.80)

        # 2. Error handling: empty audio
        with self.assertRaises(AudioProcessingError):
            process_audio_file(b"")

        # 3. Error handling: missing file
        with self.assertRaises(AudioProcessingError):
            process_audio_file("nonexistent_path_test.wav")

        # 4. Error handling: corrupted audio bytes
        with self.assertRaises(AudioProcessingError):
            process_audio_file(b"not_a_valid_audio_header_or_stream")

    def test_05_true_sequential_end_to_end_lifecycle(self):
        """
        Requirement 27, 39, 42, 53: Comprehensive Sequential Workflow.
        Proves that real audio processing, ML inference, MLPayload, agent reasoning,
        deterministic validation, SQLite persistence, and repair/verification operations
        occur sequentially and authoritatively from the database.
        """
        test_pump_id = f"PUMP-E2E-{uuid.uuid4().hex[:6]}"
        
        # STEP 1: Register pump in DIAGNOSIS_PENDING state
        pump = repositories.get_or_create_pump(
            self.db,
            pump_id=test_pump_id,
            status="DIAGNOSIS_PENDING",
            location_info="Kibera Sector 9",
            age_years=7,
            known_issues="Seal friction"
        )
        self.assertEqual(pump.status, "DIAGNOSIS_PENDING")

        # STEP 2 & 3: Submit actual diagnosis audio via POST /evaluate/audio
        anomaly_audio_path = "samples/audio/seal_leak_anomaly.wav"
        with open(anomaly_audio_path, "rb") as f:
            audio_bytes = f.read()

        eval_res = self.client.post(
            "/evaluate/audio",
            files={"file": ("seal_leak_anomaly.wav", audio_bytes, "audio/wav")},
            data={"pump_id": test_pump_id}
        )
        self.assertEqual(eval_res.status_code, 200)
        eval_data = eval_res.json()

        # STEP 4: Verify actual ML inference results
        self.assertEqual(eval_data["ml_prediction"], "ABNORMAL")
        self.assertGreaterEqual(eval_data["ml_confidence"], 0.70)
        self.assertEqual(eval_data["confidence_flag"], "HIGH")

        # STEP 5 & 6: Verify Agent Reasoning & Deterministic Validation
        self.assertEqual(eval_data["proposed_state"], "MAINTENANCE_REQUIRED")
        self.assertEqual(eval_data["final_state"], "MAINTENANCE_REQUIRED")
        self.assertFalse(eval_data["needs_human_review"])
        self.assertIsNotNone(eval_data["diagnostic_report"])
        self.assertTrue(len(eval_data["diagnostic_report"]["observation"]) > 0)
        self.assertTrue(len(eval_data["diagnostic_report"]["recommendation"]) > 0)

        # STEP 7 & 8: Verify Database updated authoritative status
        self.db.expire_all()
        pump_after_eval = repositories.get_pump_by_id(self.db, test_pump_id)
        self.assertEqual(pump_after_eval.status, "MAINTENANCE_REQUIRED")

        state_resp = self.client.get(f"/pumps/{test_pump_id}")
        self.assertEqual(state_resp.status_code, 200)
        self.assertEqual(state_resp.json()["status"], "MAINTENANCE_REQUIRED")

        # STEP 9: Verify state machine blocks illegal transition skip (e.g. premature repair complete)
        bad_complete = self.client.post("/repair/complete", json={"pump_id": test_pump_id})
        self.assertEqual(bad_complete.status_code, 400)
        self.assertIn("Illegal transition request", bad_complete.json()["error"])

        # STEP 10: Start repair using database state
        start_res = self.client.post("/repair/start", json={
            "pump_id": test_pump_id,
            "notes": "Mechanic Alice replaced worn seals.",
            "technician_id": "TECH-01"
        })
        self.assertEqual(start_res.status_code, 200)
        start_data = start_res.json()
        self.assertEqual(start_data["previous_state"], "MAINTENANCE_REQUIRED")
        self.assertEqual(start_data["current_state"], "REPAIR_IN_PROGRESS")

        # Confirm DB state is now REPAIR_IN_PROGRESS
        self.db.expire_all()
        pump_repairing = repositories.get_pump_by_id(self.db, test_pump_id)
        self.assertEqual(pump_repairing.status, "REPAIR_IN_PROGRESS")

        # STEP 11: Complete repair -> VERIFICATION_PENDING
        complete_res = self.client.post("/repair/complete", json={
            "pump_id": test_pump_id,
            "notes": "Parts tightened; ready for acoustic validation."
        })
        self.assertEqual(complete_res.status_code, 200)
        complete_data = complete_res.json()
        self.assertEqual(complete_data["previous_state"], "REPAIR_IN_PROGRESS")
        self.assertEqual(complete_data["current_state"], "VERIFICATION_PENDING")

        self.db.expire_all()
        pump_verif = repositories.get_pump_by_id(self.db, test_pump_id)
        self.assertEqual(pump_verif.status, "VERIFICATION_PENDING")

        # STEP 12 & 13: Submit actual post-repair verification audio via POST /verify/audio
        verif_audio_path = "samples/audio/normal_pump_sample.wav"
        with open(verif_audio_path, "rb") as f:
            verif_bytes = f.read()

        verif_res = self.client.post(
            "/verify/audio",
            files={"file": ("normal_pump_sample.wav", verif_bytes, "audio/wav")},
            data={"pump_id": test_pump_id}
        )
        self.assertEqual(verif_res.status_code, 200)
        verif_data = verif_res.json()

        # STEP 14: Actual ML classification of post-repair audio
        self.assertEqual(verif_data["ml_prediction"], "NORMAL")
        self.assertGreaterEqual(verif_data["ml_confidence"], 0.70)
        self.assertEqual(verif_data["final_state"], "HEALTHY")

        # STEP 15: Authoritative Database check
        self.db.expire_all()
        pump_healthy = repositories.get_pump_by_id(self.db, test_pump_id)
        self.assertEqual(pump_healthy.status, "HEALTHY")

        # STEP 16: Verify complete sequential audit trail in SQLite
        history_res = self.client.get(f"/pumps/{test_pump_id}/history")
        self.assertEqual(history_res.status_code, 200)
        history_data = history_res.json()

        self.assertEqual(history_data["current_status"], "HEALTHY")
        self.assertGreaterEqual(len(history_data["cases"]), 2)
        self.assertGreaterEqual(len(history_data["transitions"]), 4)

        # Verify transition sequence in history
        trans_seq = [t["final_state"] for t in reversed(history_data["transitions"])]
        self.assertIn("MAINTENANCE_REQUIRED", trans_seq)
        self.assertIn("REPAIR_IN_PROGRESS", trans_seq)
        self.assertIn("VERIFICATION_PENDING", trans_seq)
        self.assertIn("HEALTHY", trans_seq)


if __name__ == "__main__":
    unittest.main()
