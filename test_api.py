import unittest
from fastapi.testclient import TestClient
from api import app

class TestEmeraldAgentAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("service"), "Emerald Boys AI Agent")
        self.assertEqual(data.get("version"), "2.0")
        self.assertEqual(data.get("status"), "running")

    def test_02_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("service"), "emerald-boys-agent")
        self.assertEqual(data.get("version"), "2.0")

    def test_03_evaluate_invalid_confidence_high(self):
        payload = {
            "case_id": "TEST-422-CONF",
            "pump_id": "PUMP-001",
            "current_state": "DIAGNOSIS_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 1.5
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_04_evaluate_invalid_confidence_low(self):
        payload = {
            "case_id": "TEST-422-LOW",
            "pump_id": "PUMP-001",
            "current_state": "DIAGNOSIS_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": -0.05
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_05_evaluate_invalid_state(self):
        payload = {
            "case_id": "TEST-422-STATE",
            "pump_id": "PUMP-001",
            "current_state": "INVALID_STATE",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.85
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_06_evaluate_missing_fields(self):
        payload = {
            "case_id": "TEST-422-MISSING",
            "pump_id": "PUMP-001"
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_07_evaluate_valid_case_101(self):
        payload = {
            "case_id": "CASE-101",
            "pump_id": "PUMP-001",
            "current_state": "DIAGNOSIS_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.92
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["case_id"], "CASE-101")
        self.assertEqual(data["pump_id"], "PUMP-001")
        self.assertEqual(data["current_state"], "DIAGNOSIS_PENDING")
        self.assertEqual(data["proposed_state"], "MAINTENANCE_REQUIRED")
        self.assertEqual(data["confidence_flag"], "HIGH")
        self.assertFalse(data["needs_human_review"])
        self.assertTrue(len(data["explanation"]) > 0)

    def test_08_evaluate_unknown_pump(self):
        payload = {
            "case_id": "CASE-104",
            "pump_id": "PUMP-999",
            "current_state": "DIAGNOSIS_PENDING",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.85
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["case_id"], "CASE-104")
        self.assertEqual(data["action"], "ESCALATE")
        self.assertEqual(data["proposed_state"], "ESCALATED")
        self.assertTrue(data["needs_human_review"])

    def test_09_evaluate_illegal_transition_override(self):
        payload = {
            "case_id": "CASE-105",
            "pump_id": "PUMP-001",
            "current_state": "HEALTHY",
            "ml_prediction": "ABNORMAL",
            "ml_confidence": 0.95
        }
        response = self.client.post("/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["case_id"], "CASE-105")
        self.assertEqual(data["proposed_state"], "MAINTENANCE_REQUIRED")
        self.assertEqual(data["final_state"], "ESCALATED")
        self.assertEqual(data["action"], "ESCALATE")
        self.assertTrue(data["needs_human_review"])
        self.assertTrue(data["was_overridden"])
        self.assertIn("[OVERRIDE]", data["audit_notes"])

if __name__ == "__main__":
    unittest.main()
