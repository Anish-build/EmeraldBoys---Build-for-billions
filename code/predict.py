"""
Jal-Setu Machine Learning Module — Inference Engine
Predicts Handpump Audio Condition (normal vs abnormal) for the Agent Module.

Strict Contract Compliance:
- Loads the saved pipeline artifact from disk (no retraining).
- Validates input features against training schema.
- Reorders features strictly according to training order.
- Computes predicted class and confidence (predicted class probability).
- Outputs an Agent-compliant, JSON-serializable dictionary.
- Ground truth labels are strictly excluded from output.
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import numpy as np
import pandas as pd
import joblib

# Silence scikit-learn warnings during inference
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


class JalSetuPredictor:
    """
    Inference handler for Jal-Setu pump diagnosis models.
    Loads trained artifact and provides Agent-compatible predictions.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initializes the predictor by loading the trained model artifact.
        
        Args:
            model_path: Path to the .pkl artifact. If None, auto-resolves to
                        models/final_model.pkl if present, else models/dummy_test_model.pkl.
        """
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"

        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            final_path = models_dir / "final_model.pkl"
            dummy_path = models_dir / "dummy_test_model.pkl"
            if final_path.exists():
                self.model_path = final_path
            elif dummy_path.exists():
                self.model_path = dummy_path
            else:
                raise FileNotFoundError(
                    f"No trained model artifact found in {models_dir}. "
                    "Expected 'final_model.pkl' or 'dummy_test_model.pkl'."
                )

        self._load_artifact()

    def _load_artifact(self) -> None:
        """Loads and validates the artifact contents from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path.resolve()}")

        try:
            artifact = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifact at {self.model_path}: {e}") from e

        if not isinstance(artifact, dict) or "pipeline" not in artifact or "feature_names" not in artifact:
            raise ValueError(
                f"Invalid artifact format in {self.model_path}. "
                "Expected dict containing 'pipeline' and 'feature_names'."
            )

        self.pipeline = artifact["pipeline"]
        self.feature_names: List[str] = list(artifact["feature_names"])
        self.model_name: str = artifact.get("model_name", "UnknownModel")
        self.classes: List[str] = [str(c) for c in artifact.get("classes", self.pipeline.classes_)]

    def predict(
        self,
        features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        recording_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs inference on a single audio recording's acoustic features.
        
        Args:
            features: Dictionary, Series, or single-row DataFrame of numerical features.
            recording_id: Optional identifier (filename / recording ID).
                          If omitted, extracted from 'filename' or 'recording_id' if present in input.
                          
        Returns:
            Agent-compatible dictionary:
            {
                "recording_id": "...",
                "prediction": "normal" | "abnormal",
                "confidence": 0.xx,
                "class_probabilities": {
                    "normal": 0.xx,
                    "abnormal": 0.xx
                }
            }
        """
        # Recognized metadata identifier keys
        id_keys = ("recording_id", "filename", "file_name")

        # Convert input to DataFrame
        if isinstance(features, dict):
            features_dict = dict(features)
            # Support nested {"recording_id": "...", "features": {...}} input format per Section 30
            if "features" in features_dict and isinstance(features_dict["features"], dict):
                if recording_id is None:
                    for k in id_keys:
                        if k in features_dict:
                            recording_id = str(features_dict[k])
                            break
                    if recording_id is None:
                        recording_id = "unknown_recording.wav"
                features_dict = dict(features_dict["features"])
            else:
                if recording_id is None:
                    for k in id_keys:
                        if k in features_dict:
                            recording_id = str(features_dict.pop(k))
                            break
                    if recording_id is None:
                        recording_id = "unknown_recording.wav"
                else:
                    for k in id_keys:
                        features_dict.pop(k, None)

            # Strip any remaining metadata keys and ground-truth label
            for k in id_keys:
                features_dict.pop(k, None)
            features_dict.pop("label", None)
            df_input = pd.DataFrame([features_dict])

        elif isinstance(features, pd.Series):
            features_dict = features.to_dict()
            if recording_id is None:
                for k in id_keys:
                    if k in features_dict:
                        recording_id = str(features_dict.pop(k))
                        break
                if recording_id is None:
                    recording_id = "unknown_recording.wav"
            else:
                for k in id_keys:
                    features_dict.pop(k, None)

            for k in id_keys:
                features_dict.pop(k, None)
            features_dict.pop("label", None)
            df_input = pd.DataFrame([features_dict])

        elif isinstance(features, pd.DataFrame):
            df_input = features.copy()
            if recording_id is None:
                for k in id_keys:
                    if k in df_input.columns:
                        recording_id = str(df_input[k].iloc[0])
                        df_input = df_input.drop(columns=[k])
                        break
                if recording_id is None:
                    recording_id = "unknown_recording.wav"
            else:
                for k in id_keys:
                    if k in df_input.columns:
                        df_input = df_input.drop(columns=[k])

            # Drop any remaining id keys and label
            for k in id_keys:
                if k in df_input.columns:
                    df_input = df_input.drop(columns=[k])
            if "label" in df_input.columns:
                df_input = df_input.drop(columns=["label"])
        else:
            raise TypeError(f"Unsupported features type: {type(features)}. Expected dict, Series, or DataFrame.")

        # Validate that all required training features are present (no missing features)
        input_columns = set(df_input.columns)
        required_columns = set(self.feature_names)
        missing_columns = required_columns - input_columns
        unexpected_columns = input_columns - required_columns

        if missing_columns:
            raise ValueError(
                f"Feature validation error: missing {len(missing_columns)} required features: "
                f"{sorted(list(missing_columns))[:5]}..."
            )

        # Reject unexpected acoustic features per Sections 12 & 31
        if unexpected_columns:
            raise ValueError(
                f"Feature validation error: unexpected acoustic features detected: "
                f"{sorted(list(unexpected_columns))[:5]}..."
            )

        # Enforce exact training feature order
        X_ordered = df_input[self.feature_names].copy()

        # Validate numerical types and absence of NaNs / Infs
        for col in self.feature_names:
            try:
                X_ordered[col] = pd.to_numeric(X_ordered[col], errors="raise")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Feature validation error: column '{col}' contains non-numeric data: {e}")
            if X_ordered[col].isnull().any():
                raise ValueError(f"Feature validation error: column '{col}' contains NaN values.")
            if np.isinf(X_ordered[col]).any():
                raise ValueError(f"Feature validation error: column '{col}' contains infinite values.")

        # Run prediction through saved pipeline
        y_pred = self.pipeline.predict(X_ordered)[0]
        y_proba = self.pipeline.predict_proba(X_ordered)[0]

        # Extract class probabilities mapped to class names
        prob_dict = {
            str(cls_name): round(float(prob), 4)
            for cls_name, prob in zip(self.classes, y_proba)
        }

        # Confidence is explicitly defined as the predicted class's probability (Section 33 & 34)
        pred_str = str(y_pred)
        confidence = prob_dict.get(pred_str, round(float(np.max(y_proba)), 4))

        # Build clean Agent-compatible JSON response
        agent_response = {
            "recording_id": str(recording_id),
            "prediction": pred_str,
            "confidence": confidence,
            "class_probabilities": prob_dict,
        }

        return agent_response


# ----------------------------------------------------------------------
# Standalone Test Execution Entrypoint (Step 13 & 14 Verification)
# ----------------------------------------------------------------------
def main():
    """Executes a standalone inference test using a sample from the REAL dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    real_csv_standard = base_dir / "data" / "jal_setu_real_audio_features.csv"
    real_csv_alt = base_dir / "data" / "jal_setu_real_audio_features.csv.csv"
    real_csv = real_csv_standard if real_csv_standard.exists() else real_csv_alt

    # 1. Initialize predictor (auto-resolves to models/final_model.pkl)
    predictor = JalSetuPredictor()

    # 2. Load test sample from real data
    df = pd.read_csv(real_csv)
    sample_row = df.iloc[0].to_dict()

    sample_id = sample_row.get(
        "recording_id",
        sample_row.get("filename", sample_row.get("file_name", "sample_recording.wav"))
    )
    ground_truth = sample_row.get("label", "unknown")

    # 3. Strip ground-truth label to verify inference independence
    features_only = {k: v for k, v in sample_row.items() if k not in ("label",)}

    # 4. Execute prediction
    agent_output = predictor.predict(features=features_only, recording_id=sample_id)

    # 5. Format terminal output matching Section 32
    print("=" * 60)
    print("JAL-SETU ML INFERENCE")
    print("=" * 60)
    print(f"\nModel: {predictor.model_path.name}")
    print(f"Recording: {sample_id}")
    print(f"Prediction: {agent_output['prediction']}")
    print(f"Confidence: {agent_output['confidence']}")
    print("\nAgent Output:")
    print(json.dumps(agent_output, indent=2))
    print("=" * 60)

    # 6. Integrity checks
    assert "label" not in agent_output, "Ground truth 'label' leaked into Agent output!"
    assert "prediction" in agent_output, "Missing 'prediction' key in Agent output!"
    assert "confidence" in agent_output, "Missing 'confidence' key in Agent output!"
    assert "class_probabilities" in agent_output, "Missing 'class_probabilities' key in Agent output!"
    assert agent_output["prediction"] in predictor.classes, "Predicted class not in known classes!"
    assert abs(sum(agent_output["class_probabilities"].values()) - 1.0) < 1e-3, "Probabilities do not sum to 1.0!"


if __name__ == "__main__":
    main()
