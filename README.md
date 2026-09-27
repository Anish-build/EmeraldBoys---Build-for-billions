# Jal-Setu — Machine Learning Module

## 1. Purpose & Architecture
**Jal-Setu** is an agentic AI system for rural handpump maintenance.
The end-to-end system flow is:
```text
DATA → AUDIO → ML → AGENT → DATABASE / DASHBOARD
```
The **ML module** is responsible strictly for **binary classification**: diagnosing whether an audio recording of a handpump indicates a `normal` or `abnormal` mechanical condition.

---

## 2. Team Responsibility Boundaries
* **DATA**: Raw video/audio recordings and ground-truth metadata.
* **AUDIO**: Audio extraction, resampling, mono conversion, silence trimming, volume normalization, denoising, and acoustic feature extraction (MFCCs, ZCR, Spectral Centroid, Bandwidth, Rolloff, Mel-derived, Chroma). Audio outputs the feature CSV.
* **ML (This Module)**: Consumes the acoustic feature CSV, validates data integrity, trains/evaluates classical candidate models, exports model pipelines, and delivers inference predictions to the Agent. ML does **not** process raw audio or use `librosa`.
* **AGENT**: Receives diagnosis & confidence from ML, manages dispatch workflows, severity decisions, human reviews, mechanic assignment, and verification.
* **DATABASE & DASHBOARD**: Persistence and operational visualization.

---

## 3. Dataset Format & Schema Rules
* **Input CSV Format**:
  ```text
  file_name / filename | acoustic_feature_1 | acoustic_feature_2 | ... | label
  ```
* **Target Classes**: Binary classification (`normal` vs `abnormal`).
* **$X$ and $y$**:
  * $X$: Purely numerical acoustic features (dynamically identified after dropping recognized metadata like `file_name`/`filename`/`recording_id`, grouping columns, and the target `label`).
  * $y$: Ground truth target label (`normal` or `abnormal`). True $y$ is strictly unknown during inference.
* **Dynamic Feature & Sample Count**: The ML pipeline dynamically determines the number and names of acoustic features, as well as the dataset row count. Feature counts and row counts are **never hard-coded**. (Expected real dataset: 500 samples — 250 normal, 250 abnormal).
* **Source / Recording Leakage**: Metadata is inspected for grouping columns (e.g. `pump_id`, `location_id`). If absent, the limitation is documented and standard stratified split is applied; real-world deployment requires environmental and pump diversity.
* **Strict Schema Validation**:
  * Missing features $\rightarrow$ validation error.
  * Unexpected acoustic features $\rightarrow$ validation error.
  * Same features in different column order $\rightarrow$ automatically reordered to match training order.
  * Non-numeric / NaN / Infinite values $\rightarrow$ validation error.

---

## 4. Training Pipeline (`code/ml_pipeline.py`)
* **Stratified Train / Test Split**: 80% training / 20% held-out test set (`random_state=42`). The test set remains completely untouched until final evaluation.
* **Small-Dataset Safeguard**: Dynamically warns when small datasets (< 50 samples) are detected, since metrics on very small partitions can exhibit high statistical variance.
* **Cross-Validation**: `StratifiedKFold` (5 folds, dynamically adapted to minority class count) performed strictly on the training partition.
* **Leakage-Safe Scaling**: `StandardScaler` is wrapped inside scikit-learn `Pipeline` objects for Logistic Regression and SVM, fitting strictly on training folds. Random Forest is not scaled.
* **Candidate Models**:
  1. **Logistic Regression**: with `StandardScaler`, `max_iter=2000`, `class_weight='balanced'`. Tuned across `C`.
  2. **Random Forest**: `n_estimators=300`, `class_weight='balanced'`. Tuned across `n_estimators`, `max_depth`, `min_samples_split`.
  3. **RBF SVM**: with `StandardScaler`, `probability=True`, `class_weight='balanced'`. Tuned across `C` and `gamma`.
* **Hyperparameter Tuning**: `GridSearchCV` on the training split using abnormal-class $F_1$ as the primary optimization metric.
* **Zero-Leakage Model Selection**: Objective ranking based strictly on Training Cross-Validation results (Mean CV $F_1$, Mean CV Recall on abnormal class, CV stability across folds, and parsimony tie-breaking). Test set metrics are completely excluded from model selection.
* **Post-Selection Test Evaluation**: Only AFTER the final model has been selected is it evaluated exactly once on the untouched 20% held-out test set.
* **Evaluation Metrics**: Accuracy, Precision, Abnormal Recall (maintenance critical: minimizes missed abnormal pumps / FN), $F_1$-score, Confusion Matrix, ROC-AUC, and PR-AUC.

---

## 5. Model Artifacts: Dummy vs. Real
* **Dummy Test Model**: [`models/dummy_test_model.pkl`](models/dummy_test_model.pkl) — generated during dummy pipeline testing. Never treated as real performance evidence.
* **Real Production Model**: [`models/final_model.pkl`](models/final_model.pkl) — generated strictly when the real Audio team CSV is provided.
* The saved artifact is a bundle containing the fitted pipeline, exact feature ordering metadata, and target classes.

---

## 6. Inference Engine (`code/predict.py`)
`code/predict.py` is **inference-only**:
* Loads the saved model artifact from disk.
* **Never retrains** or runs cross-validation.
* Does **not** require ground-truth labels.
* Accepts single-record dictionaries, Series, or DataFrames (including nested `{"recording_id": "...", "features": {...}}`).
* Validates schema, reorders columns, and evaluates probability distributions.

---

## 7. ML → Agent Output Contract
The Agent module receives a JSON-serializable dictionary:
```json
{
  "recording_id": "recording_01.wav",
  "prediction": "abnormal",
  "confidence": 0.91,
  "class_probabilities": {
    "normal": 0.09,
    "abnormal": 0.91
  }
}
```
* **Confidence**: Defined as the probability/score associated with the predicted class (it indicates model probability, not absolute real-world certainty).
* Ground-truth labels and internal evaluation matrices are strictly excluded.

---

## 8. Execution Commands
Using the project's virtual environment:
```powershell
# 1. Run ML Training Pipeline
.\.venv\Scripts\python.exe code/ml_pipeline.py

# 2. Run Inference Engine Test
.\.venv\Scripts\python.exe code/predict.py
```

---

## 9. Audio → ML Handoff Protocol
When the real Audio feature CSV arrives:
1. Place CSV in `data/` (e.g. `data/jal_setu_real_audio_features.csv`).
2. Set `RUN_MODE = "REAL"` in `PipelineConfig` within `code/ml_pipeline.py`.
3. Execute `ml_pipeline.py` to train completely from scratch, producing `models/final_model.pkl`.
4. Inference via `predict.py` will automatically resolve and utilize `models/final_model.pkl`.
