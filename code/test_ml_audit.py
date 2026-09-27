"""
Comprehensive Audit Test Suite for Jal-Setu ML Module
Executes and validates all required schema tests (A through M).
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure code directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "code"))

from predict import JalSetuPredictor
from ml_pipeline import PipelineConfig


def run_all_audit_tests():
    print("=" * 70)
    print("JAL-SETU ML MODULE — COMPREHENSIVE AUDIT TEST SUITE (Tests A through M)")
    print("=" * 70)

    dummy_csv = BASE_DIR / "data" / "jal_setu_dummy_audio_features.csv"
    dummy_model = BASE_DIR / "models" / "dummy_test_model.pkl"
    final_model = BASE_DIR / "models" / "final_model.pkl"

    df = pd.read_csv(dummy_csv)
    normal_sample = df[df["label"] == "normal"].iloc[0].to_dict()
    abnormal_sample = df[df["label"] == "abnormal"].iloc[0].to_dict()

    predictor = JalSetuPredictor(model_path=dummy_model)
    feature_names = predictor.feature_names

    # Test A: VALID NORMAL INPUT
    print("\n[TEST A] Valid Normal Input:")
    norm_features = {k: v for k, v in normal_sample.items() if k not in ("label",)}
    res_a = predictor.predict(norm_features, recording_id="normal_test.wav")
    assert res_a["prediction"] == "normal", f"Expected normal, got {res_a['prediction']}"
    assert "label" not in res_a
    print(f"  [PASS] Prediction: '{res_a['prediction']}', Confidence: {res_a['confidence']}")

    # Test B: VALID ABNORMAL INPUT
    print("\n[TEST B] Valid Abnormal Input:")
    abnorm_features = {k: v for k, v in abnormal_sample.items() if k not in ("label",)}
    res_b = predictor.predict(abnorm_features, recording_id="abnormal_test.wav")
    assert res_b["prediction"] == "abnormal", f"Expected abnormal, got {res_b['prediction']}"
    assert "label" not in res_b
    print(f"  [PASS] Prediction: '{res_b['prediction']}', Confidence: {res_b['confidence']}")

    # Test C: SAME FEATURES, DIFFERENT ORDER
    print("\n[TEST C] Same Features, Different Order (Reordering Check):")
    reversed_features = {k: norm_features[k] for k in reversed(list(norm_features.keys()))}
    res_c = predictor.predict(reversed_features, recording_id="reversed_test.wav")
    assert res_c["prediction"] == "normal"
    assert res_c["confidence"] == res_a["confidence"], "Confidence mismatch on reordered features!"
    print(f"  [PASS] Automatically reordered and matched prediction (Confidence: {res_c['confidence']})")

    # Test D: MISSING FEATURE
    print("\n[TEST D] Missing Feature Schema Rejection:")
    missing_features = norm_features.copy()
    omitted_col = feature_names[0]
    missing_features.pop(omitted_col)
    try:
        predictor.predict(missing_features)
        raise AssertionError("Failed: predict() did not reject missing feature!")
    except ValueError as e:
        print(f"  [PASS] Successfully rejected missing feature: {e}")

    # Test E: UNEXPECTED ACOUSTIC FEATURE
    print("\n[TEST E] Unexpected Acoustic Feature Schema Rejection:")
    extra_features = norm_features.copy()
    extra_features["bogus_acoustic_energy"] = 999.99
    try:
        predictor.predict(extra_features)
        raise AssertionError("Failed: predict() did not reject unexpected feature!")
    except ValueError as e:
        print(f"  [PASS] Successfully rejected unexpected feature: {e}")

    # Test F: NON-NUMERIC FEATURE VALUE
    print("\n[TEST F] Non-Numeric Feature Value Rejection:")
    corrupt_features = norm_features.copy()
    corrupt_features[feature_names[0]] = "corrupt_string_value"
    try:
        predictor.predict(corrupt_features)
        raise AssertionError("Failed: predict() did not reject non-numeric feature!")
    except ValueError as e:
        print(f"  [PASS] Successfully rejected non-numeric feature: {e}")

    # Test G: NO GROUND-TRUTH LABEL
    print("\n[TEST G] Inference with No Ground-Truth Label:")
    res_g = predictor.predict(norm_features)
    assert "label" not in res_g
    assert "prediction" in res_g
    assert "confidence" in res_g
    print(f"  [PASS] Successfully predicted without ground-truth label: {res_g['prediction']}")

    # Test H: PREDICT.PY DOES NOT RETRAIN
    print("\n[TEST H] predict.py Inference-Only Verification (No Retraining):")
    # Verify model weights/file timestamp do not change during predict
    mtime_before = dummy_model.stat().st_mtime
    predictor.predict(norm_features)
    mtime_after = dummy_model.stat().st_mtime
    assert mtime_before == mtime_after, "Model file was modified during inference!"
    assert not hasattr(predictor, "fit"), "Predictor object should not expose fit()!"
    print("  [PASS] Confirmed predict.py is strictly inference-only.")

    # Test I: MODEL ARTIFACT SEPARATION (POST-TRAINING)
    print("\n[TEST I] Model Artifact Separation (Dummy vs Real):")
    import joblib
    assert final_model.exists(), "models/final_model.pkl should exist!"
    assert dummy_model.exists(), "models/dummy_test_model.pkl should exist!"
    assert final_model.resolve() != dummy_model.resolve(), "The two artifact paths must be distinct!"
    assert final_model.stat().st_size != dummy_model.stat().st_size, "The two artifact files must not be the same file!"

    # Verify both artifacts load successfully
    real_artifact = joblib.load(final_model)
    dummy_artifact = joblib.load(dummy_model)
    assert isinstance(real_artifact, dict) and "pipeline" in real_artifact, "Failed to load real model artifact!"
    assert isinstance(dummy_artifact, dict) and "pipeline" in dummy_artifact, "Failed to load dummy model artifact!"

    # Verify real artifact is the trained Random Forest model and dummy artifact remains separate
    assert real_artifact.get("model_name") == "Random Forest", f"Expected real model to be 'Random Forest', got {real_artifact.get('model_name')}"
    assert dummy_artifact.get("model_name") == "Logistic Regression", f"Expected dummy model to be 'Logistic Regression', got {dummy_artifact.get('model_name')}"
    assert real_artifact["feature_names"] != dummy_artifact["feature_names"], "Real and dummy feature sets must remain distinct!"
    print(f"  [PASS] Verified distinct artifacts: real '{real_artifact['model_name']}' ({len(real_artifact['feature_names'])} features) and dummy '{dummy_artifact['model_name']}' ({len(dummy_artifact['feature_names'])} features).")

    # Test J: FEATURE SCHEMA IN MODEL
    print("\n[TEST J] Feature Schema Stored in Model Artifact:")
    meta_cols = [c for c in ["filename", "file_name", "recording_id", "label"] if c in df.columns]
    expected_feats = list(df.drop(columns=meta_cols).columns)
    assert len(predictor.feature_names) == len(expected_feats), f"Expected {len(expected_feats)} features, found {len(predictor.feature_names)}"
    assert predictor.feature_names == expected_feats
    print(f"  [PASS] Model artifact preserves exact dynamic schema ({len(predictor.feature_names)} features in order).")

    # Test K: RELATIVE PATHS & GITHUB PORTABILITY
    print("\n[TEST K] Project-Relative Paths (No Machine-Specific Paths in Production Code):")
    production_files = ["ml_pipeline.py", "predict.py", "__init__.py"]
    for fname in production_files:
        py_file = BASE_DIR / "code" / fname
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "C:\\Users" not in content, f"Hardcoded path found in {py_file.name}!"
            assert "c:/users" not in content.lower(), f"Hardcoded path found in {py_file.name}!"
    print("  [PASS] Zero machine-specific absolute paths in production python files.")

    # Test L: ZERO TEST LEAKAGE IN MODEL SELECTION
    print("\n[TEST L] Model Selection Test Leakage Prevention:")
    import inspect
    from ml_pipeline import select_best_model
    sig = inspect.signature(select_best_model)
    assert "eval_results" not in sig.parameters, "select_best_model still accepts eval_results (test leakage)!"
    assert "cv_results" in sig.parameters, "select_best_model must evaluate strictly from cv_results!"
    print("  [PASS] Confirmed select_best_model() signature is strictly isolated to CV results (0% test leakage).")

    # Test M: FLEXIBLE METADATA IDENTIFIER ALIAS (file_name / recording_id)
    print("\n[TEST M] Flexible Metadata Identifier Alias Handling:")
    # Create sample using only feature columns plus 'file_name' metadata alias
    file_name_sample = {k: v for k, v in normal_sample.items() if k not in ("filename", "recording_id", "file_name", "label")}
    file_name_sample["file_name"] = "pump_recording_001.wav"
    res_m = predictor.predict(file_name_sample)
    assert res_m["recording_id"] == "pump_recording_001.wav", f"Expected recording_id 'pump_recording_001.wav', got {res_m.get('recording_id')}"
    assert "prediction" in res_m and res_m["prediction"] in ("normal", "abnormal")
    assert "confidence" in res_m
    assert "file_name" not in res_m
    print(f"  [PASS] Auto-extracted recording_id from 'file_name' key: '{res_m['recording_id']}'")

    # Also verify 'recording_id' key directly
    rec_id_sample = {k: v for k, v in normal_sample.items() if k not in ("filename", "file_name", "recording_id", "label")}
    rec_id_sample["recording_id"] = "pump_recording_042.wav"
    res_m2 = predictor.predict(rec_id_sample)
    assert res_m2["recording_id"] == "pump_recording_042.wav", f"Expected recording_id 'pump_recording_042.wav', got {res_m2.get('recording_id')}"
    assert "prediction" in res_m2 and res_m2["prediction"] in ("normal", "abnormal")
    print(f"  [PASS] Auto-extracted recording_id from 'recording_id' key: '{res_m2['recording_id']}'")

    print("\n" + "=" * 70)
    print("ALL 13 AUDIT TESTS (A through M) PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_audit_tests()
