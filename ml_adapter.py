<<<<<<< HEAD
import os
=======
"""
ML Adapter Module — External ML Contract Boundary & Local Fallback
Part of Emerald Boys Architecture.

DIVISION OF RESPONSIBILITY:
The Audio & ML team owns raw audio preprocessing, feature extraction,
model training/inference, and acoustic datasets.
This adapter defines the external ML contract and provides a deterministic
MOCK ML fallback for offline development and test suites.
"""

>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
import uuid
import logging
from typing import Dict, Any, Optional
from schemas import MLPayload, PumpState

logger = logging.getLogger("EmeraldMLAdapter")

<<<<<<< HEAD
# Explicit mode identifier for transparency
ADAPTER_MODE = "MOCK ML"

# Deterministic simulated acoustic profiles for V3 mock inference
=======
# Explicit mode identifier for contract transparency
ADAPTER_MODE = "MOCK ML"

# Deterministic simulated acoustic profiles for testing and development fallback
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
AUDIO_PROFILES = {
    "normal_rhythm": {"prediction": "NORMAL", "confidence": 0.88},
    "seal_friction": {"prediction": "ABNORMAL", "confidence": 0.92},
    "bearing_chatter": {"prediction": "ABNORMAL", "confidence": 0.78},
    "valve_leak_subtle": {"prediction": "ABNORMAL", "confidence": 0.35},
    "high_confidence": {"prediction": "ABNORMAL", "confidence": 0.95},
    "low_confidence": {"prediction": "ABNORMAL", "confidence": 0.30},
}


class MLAdapter:
    """
<<<<<<< HEAD
    Adapter interfacing raw audio/sensor data pipelines with the Emerald Boys AI Agent.
    
    DISCLAIMER:
    Current implementation operates in 'MOCK ML' mode. It produces deterministic structured
    results conforming to the MLPayload schema for software prototyping and verification.
    It does NOT represent a real-world physical acoustic model.
=======
    Contract boundary interfacing external ML results with the Emerald Boys AI Agent.
    Operates in deterministic 'MOCK ML' mode as a development fallback.
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
    """
    MODE = ADAPTER_MODE

    @classmethod
    def predict(
        cls,
        audio_path: str,
        pump_id: str = "PUMP-001",
        current_state: PumpState = "DIAGNOSIS_PENDING",
        case_id: Optional[str] = None
    ) -> MLPayload:
        """
<<<<<<< HEAD
        Accepts an audio file path, executes deterministic mock acoustic classification,
        and constructs a strictly validated MLPayload.

        Parameters:
        - audio_path: str (path or filename of the audio recording)
        - pump_id: str (target pump identifier)
        - current_state: PumpState (current lifecycle state of the pump)
        - case_id: Optional unique case ID (auto-generated if omitted)
=======
        Accepts an audio file path or sample key, executes deterministic mock classification,
        and constructs a strictly validated MLPayload conforming to the external ML contract.
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
        """
        if not audio_path or not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError("Audio path must be a non-empty string.")

        clean_path = audio_path.lower().strip()
        cid = case_id or f"CASE-ML-{uuid.uuid4().hex[:8]}"

        # Deterministic profile classification based on filename/path keywords
        if "normal" in clean_path or "healthy" in clean_path:
            prediction = "NORMAL"
            confidence = 0.88
        elif "seal" in clean_path or "friction" in clean_path or "fault" in clean_path or "high_confidence" in clean_path:
            prediction = "ABNORMAL"
            confidence = 0.92
        elif "bearing" in clean_path or "medium" in clean_path:
            prediction = "ABNORMAL"
            confidence = 0.78
        elif "leak" in clean_path or "subtle" in clean_path or "low_confidence" in clean_path or "uncertain" in clean_path:
            prediction = "ABNORMAL"
            confidence = 0.35
        elif "abnormal" in clean_path:
            prediction = "ABNORMAL"
            confidence = 0.92
        else:
            prediction = "ABNORMAL"
            confidence = 0.85

        logger.info(
<<<<<<< HEAD
            f"[{cls.MODE}] Processed audio '{audio_path}' -> {prediction} (Confidence: {confidence:.2f})"
=======
            f"[{cls.MODE}] Ingested audio reference '{audio_path}' -> {prediction} (Confidence: {confidence:.2f})"
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
        )

        return MLPayload(
            case_id=cid,
            pump_id=str(pump_id),
            current_state=current_state,
            ml_prediction=prediction,
            ml_confidence=confidence
        )

    @classmethod
    def run_inference(cls, input_data: Dict[str, Any]) -> MLPayload:
        """
<<<<<<< HEAD
        Executes deterministic feature inference and constructs a strictly validated MLPayload.

        Expected input_data keys:
        - pump_id: str (required)
        - current_state: PumpState (required)
        - case_id: Optional[str] (generated if omitted)
        - audio_profile: Optional[str] (key from AUDIO_PROFILES)
        - audio_file_path: Optional[str] (mock file path)
=======
        Constructs a strictly validated MLPayload from a dictionary payload.
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary.")

        pump_id = input_data.get("pump_id")
        if not pump_id or not isinstance(pump_id, str):
            raise ValueError("Missing or invalid required parameter 'pump_id'.")

        current_state = input_data.get("current_state")
        if not current_state:
            raise ValueError("Missing required parameter 'current_state'.")

        case_id = input_data.get("case_id") or f"CASE-ML-{uuid.uuid4().hex[:8]}"
<<<<<<< HEAD

        audio_profile = input_data.get("audio_profile")
        audio_file_path = str(input_data.get("audio_file_path", "")).lower()

        # Deterministic feature classification
=======
        audio_profile = input_data.get("audio_profile")
        audio_file_path = str(input_data.get("audio_file_path", "")).strip()

>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
        if audio_profile in AUDIO_PROFILES:
            profile_data = AUDIO_PROFILES[audio_profile]
            prediction = profile_data["prediction"]
            confidence = profile_data["confidence"]
        elif audio_file_path:
<<<<<<< HEAD
            payload = cls.predict(
=======
            return cls.predict(
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
                audio_path=audio_file_path,
                pump_id=pump_id,
                current_state=current_state,
                case_id=case_id
            )
<<<<<<< HEAD
            return payload
        else:
            # Deterministic default profile
=======
        else:
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
            prediction = "ABNORMAL"
            confidence = 0.85

        return MLPayload(
            case_id=case_id,
            pump_id=str(pump_id),
            current_state=current_state,
            ml_prediction=prediction,
            ml_confidence=confidence
        )


def predict(
    audio_path: str,
    pump_id: str = "PUMP-001",
    current_state: PumpState = "DIAGNOSIS_PENDING",
    case_id: Optional[str] = None
) -> MLPayload:
<<<<<<< HEAD
    """Clean interface exposing ML audio prediction as specified in V3 Step 3."""
=======
    """Clean interface exposing ML audio prediction boundary."""
>>>>>>> 2c902f96110e6f1f23973339b34f74bce834be2c
    return MLAdapter.predict(
        audio_path=audio_path,
        pump_id=pump_id,
        current_state=current_state,
        case_id=case_id
    )


def run_inference(input_data: Dict[str, Any]) -> MLPayload:
    """Convenience function wrapper for MLAdapter.run_inference."""
    return MLAdapter.run_inference(input_data)
