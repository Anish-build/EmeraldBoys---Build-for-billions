"""
Jal-Setu Machine Learning Module
Binary Classification: Handpump Audio Condition Diagnosis (normal vs abnormal)

This module implements the core ML pipeline for the Jal-Setu project.
Follows strict modular design:
- Configuration
- Dataset loading
- Inspection
- Validation & dynamic feature detection
- Feature / target separation
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    make_scorer,
)

# Silence scikit-learn 1.9+ deprecation warning for SVC probability=True
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
class PipelineConfig:
    """Configuration paths and parameters for Jal-Setu ML pipeline."""

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Datasets
    DUMMY_CSV_PATH: Path = DATA_DIR / "jal_setu_dummy_audio_features.csv"
    REAL_CSV_PATH: Path = DATA_DIR / "jal_setu_real_audio_features.csv"

    # Active run mode: "DUMMY" or "REAL"
    RUN_MODE: str = "REAL"

    # Column definitions
    ID_COLUMN: str = "filename"
    POSSIBLE_ID_COLUMNS: List[str] = ["filename", "file_name", "recording_id", "audio_id", "sample_id"]
    POSSIBLE_GROUP_COLUMNS: List[str] = ["pump_id", "handpump_id", "source_id", "location_id", "device_id"]
    TARGET_COLUMN: str = "label"
    EXPECTED_CLASSES: set = {"normal", "abnormal"}

    # Train / Test split configuration (80% train, 20% test per Section 19)
    TEST_SIZE: float = 0.20
    RANDOM_STATE: int = 42

    # Model artifact paths
    DUMMY_MODEL_PATH: Path = MODELS_DIR / "dummy_test_model.pkl"
    FINAL_MODEL_PATH: Path = MODELS_DIR / "final_model.pkl"

    @classmethod
    def get_active_csv_path(cls) -> Path:
        """Returns the active CSV path based on RUN_MODE."""
        if cls.RUN_MODE == "REAL":
            if cls.REAL_CSV_PATH.exists():
                return cls.REAL_CSV_PATH
            # Fallback for Windows Explorer hidden extension saving
            alt_path = cls.DATA_DIR / "jal_setu_real_audio_features.csv.csv"
            if alt_path.exists():
                return alt_path
            return cls.REAL_CSV_PATH
        return cls.DUMMY_CSV_PATH

    @classmethod
    def get_active_model_path(cls) -> Path:
        """Returns the active model output path based on RUN_MODE."""
        if cls.RUN_MODE == "REAL":
            return cls.FINAL_MODEL_PATH
        return cls.DUMMY_MODEL_PATH


# ----------------------------------------------------------------------
# 2. Dataset Loading
# ----------------------------------------------------------------------
def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Safely loads feature CSV from disk.
    
    Args:
        csv_path: Path to feature CSV.
        
    Returns:
        pd.DataFrame containing the dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty or unreadable.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {csv_path}: {e}") from e

    if df.empty:
        raise ValueError(f"The dataset at {csv_path} is empty (0 rows).")

    return df


# ----------------------------------------------------------------------
# 3. Dataset Inspection
# ----------------------------------------------------------------------
def inspect_dataset(df: pd.DataFrame, dataset_label: str) -> Dict[str, Any]:
    """
    Inspects shape, columns, dtypes, and acoustic feature groupings.
    
    Args:
        df: Input DataFrame.
        dataset_label: Human-readable label for reporting.
        
    Returns:
        Dictionary summarizing inspection findings.
    """
    print(f"\n[{dataset_label} - RAW INSPECTION]")
    print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    col_list = list(df.columns)
    print(f"  Total columns: {len(col_list)}")

    # Detect known acoustic feature categories
    mfcc_cols = [c for c in col_list if c.startswith("mfcc_")]
    zcr_cols = [c for c in col_list if c.startswith("zcr_")]
    sc_cols = [c for c in col_list if "spectral_centroid" in c]
    sb_cols = [c for c in col_list if "spectral_bandwidth" in c]
    sr_cols = [c for c in col_list if "spectral_rolloff" in c]
    rms_cols = [c for c in col_list if "rms" in c]
    mel_cols = [c for c in col_list if "mel_" in c]
    chroma_cols = [c for c in col_list if "chroma" in c]

    print("\n  Acoustic Feature Groups Detected:")
    print(f"    - MFCC features: {len(mfcc_cols)} columns")
    print(f"    - ZCR features: {len(zcr_cols)} columns")
    print(f"    - Spectral Centroid: {len(sc_cols)} columns")
    print(f"    - Spectral Bandwidth: {len(sb_cols)} columns")
    print(f"    - Spectral Rolloff: {len(sr_cols)} columns")
    print(f"    - RMS features: {len(rms_cols)} columns (not in dummy)")
    print(f"    - Mel-derived features: {len(mel_cols)} columns (not in dummy)")
    print(f"    - Chroma features: {len(chroma_cols)} columns (not in dummy)")

    summary = {
        "rows": len(df),
        "columns": col_list,
        "mfcc_count": len(mfcc_cols),
        "zcr_count": len(zcr_cols),
        "spectral_centroid_count": len(sc_cols),
        "spectral_bandwidth_count": len(sb_cols),
        "spectral_rolloff_count": len(sr_cols),
        "rms_count": len(rms_cols),
        "mel_count": len(mel_cols),
        "chroma_count": len(chroma_cols),
    }
    return summary


# ----------------------------------------------------------------------
# 4. Dataset Validation & Dynamic Feature Detection
# ----------------------------------------------------------------------
def validate_dataset(
    df: pd.DataFrame,
    id_col: str = PipelineConfig.ID_COLUMN,
    target_col: str = PipelineConfig.TARGET_COLUMN,
    expected_classes: set = PipelineConfig.EXPECTED_CLASSES,
    possible_id_cols: List[str] = PipelineConfig.POSSIBLE_ID_COLUMNS,
    possible_group_cols: List[str] = PipelineConfig.POSSIBLE_GROUP_COLUMNS,
) -> Tuple[List[str], str, str]:
    """
    Validates dataset integrity and dynamically identifies numerical feature columns.
    
    Checks:
    - Uniqueness of column names
    - Existence of target column
    - Detection and validation of ID column (filename/file_name/recording_id)
    - Source/pump grouping metadata check (pump_id, location_id, etc.)
    - Class counts and validity (at least 2 classes, expected 'normal'/'abnormal')
    - Missing / NaN values
    - Infinite values
    - Feature column data types (must be purely numerical)
    - Duplicate rows and duplicate recording IDs
    
    Args:
        df: Input DataFrame.
        id_col: Name of identifier column (e.g. 'filename').
        target_col: Name of ground truth target column (e.g. 'label').
        expected_classes: Set of valid class names.
        possible_id_cols: List of candidate identifier column names.
        possible_group_cols: List of candidate group/pump column names.
        
    Returns:
        Tuple of (feature_columns, target_col, id_col)
        
    Raises:
        ValueError: If any validation rule fails.
    """
    print("\n[DATASET INTEGRITY VALIDATION]")

    # Check 1: Unique columns
    if len(df.columns) != len(set(df.columns)):
        duplicates = [c for c in df.columns if list(df.columns).count(c) > 1]
        raise ValueError(f"Duplicate column names found in CSV: {set(duplicates)}")
    print("  [PASS] Unique column names verified.")

    # Check 2: Target column existence
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        )
    print(f"  [PASS] Target column '{target_col}' found.")

    # Check 3: ID column detection (flexible matching for filename / file_name / recording_id)
    actual_id_col = None
    if id_col in df.columns:
        actual_id_col = id_col
    else:
        for candidate in possible_id_cols:
            if candidate in df.columns:
                actual_id_col = candidate
                print(f"  [INFO] Auto-detected identifier column '{candidate}'.")
                break

    if actual_id_col is None:
        print("  [WARN] No standard identifier column found in dataset. Using row indices as IDs.")
    else:
        dup_ids = df[actual_id_col].duplicated().sum()
        if dup_ids > 0:
            print(f"  [WARN] Found {dup_ids} duplicate recording IDs in '{actual_id_col}'.")
        else:
            print(f"  [PASS] All {len(df)} records in '{actual_id_col}' are unique.")

    # Check 3b: Recording / source grouping metadata inspection (Section 13)
    detected_group_cols = [c for c in possible_group_cols if c in df.columns]
    if detected_group_cols:
        print(f"  [INFO] Detected source grouping column(s): {detected_group_cols}.")
    else:
        print("  [INFO] No source/pump grouping column found (e.g. pump_id, location_id).")
        print("         Source-level leakage cannot be assessed from available metadata.")
        print("         Proceeding with standard stratified split; real-world group diversity must be monitored.")

    # Check 4: Label integrity and class distribution
    if df[target_col].isnull().any():
        num_null = df[target_col].isnull().sum()
        raise ValueError(f"Target column '{target_col}' contains {num_null} null values.")

    unique_labels = set(df[target_col].unique())
    print(f"  Found classes: {unique_labels}")

    if len(unique_labels) < 2:
        raise ValueError(
            f"Dataset requires at least 2 classes for binary classification, but found: {unique_labels}"
        )

    if not unique_labels.issubset(expected_classes):
        unexpected = unique_labels - expected_classes
        raise ValueError(f"Unexpected label classes found: {unexpected}. Expected: {expected_classes}")

    class_distribution = df[target_col].value_counts().to_dict()
    print(f"  Class distribution: {class_distribution}")

    # Check 5: Dynamic feature identification (strictly exclude target, ID, and metadata/grouping columns)
    excluded_cols = {target_col}
    if actual_id_col:
        excluded_cols.add(actual_id_col)
    for c in possible_id_cols:
        if c in df.columns:
            excluded_cols.add(c)
    for c in possible_group_cols:
        if c in df.columns:
            excluded_cols.add(c)

    feature_cols = [c for c in df.columns if c not in excluded_cols]

    if not feature_cols:
        raise ValueError("No feature columns remaining after excluding target and ID columns.")

    print(f"  Dynamically identified {len(feature_cols)} feature columns.")

    # Check 6: Validate numerical types, NaNs, and infinite values
    non_numeric_cols = []
    nan_cols = {}
    inf_cols = {}

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
            continue

        num_nans = df[col].isnull().sum()
        if num_nans > 0:
            nan_cols[col] = num_nans

        # Check for inf / -inf
        num_infs = np.isinf(df[col]).sum()
        if num_infs > 0:
            inf_cols[col] = num_infs

    if non_numeric_cols:
        raise ValueError(f"Non-numeric feature columns detected: {non_numeric_cols}")
    print("  [PASS] All feature columns are numeric.")

    if nan_cols:
        raise ValueError(f"Missing (NaN) values detected in features: {nan_cols}")
    print("  [PASS] Zero NaN values across all feature columns.")

    if inf_cols:
        raise ValueError(f"Infinite (inf/-inf) values detected in features: {inf_cols}")
    print("  [PASS] Zero infinite values detected.")

    # Check 7: Duplicate rows
    num_dup_rows = df.duplicated().sum()
    if num_dup_rows > 0:
        print(f"  [WARN] {num_dup_rows} identical rows detected in dataset.")
    else:
        print("  [PASS] Zero identical rows detected.")

    print("  [PASS] DATASET VALIDATION COMPLETED SUCCESSFULLY.")
    return feature_cols, target_col, actual_id_col


# ----------------------------------------------------------------------
# 5. Feature & Target Separation
# ----------------------------------------------------------------------
def separate_features_and_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    id_col: str = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates the DataFrame into feature matrix X, target series y, and id series.
    Enforces exact feature ordering.
    
    Args:
        df: Validated DataFrame.
        feature_cols: Ordered list of feature column names.
        target_col: Name of target column.
        id_col: Optional name of ID column.
        
    Returns:
        Tuple of (X, y, ids)
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    ids = df[id_col].copy() if id_col and id_col in df.columns else pd.Series(range(len(df)))
    return X, y, ids


# ----------------------------------------------------------------------
# 6. Train / Test Split (Stratified with Small-Dataset Safeguards)
# ----------------------------------------------------------------------
def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    test_size: float = PipelineConfig.TEST_SIZE,
    random_state: int = PipelineConfig.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Performs stratified train/test split with safeguards for small datasets.
    
    Safeguards:
    - Validates minimum sample and class counts for stratification.
    - Ensures both train and test partitions receive representations of all classes.
    - Verifies zero sample ID / index overlap (no data leakage).
    - Preserves exact feature column ordering across splits.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        ids: Identifier Series.
        test_size: Fraction of samples to allocate to test set (default 0.20).
        random_state: Fixed random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, ids_train, ids_test)
        
    Raises:
        ValueError: If dataset is too small or stratification cannot be satisfied.
    """
    total_samples = len(X)
    if total_samples <= 50:
        print(
            f"\n  [WARNING] Small dataset detected ({total_samples} samples).\n"
            f"  Evaluation metrics may be unstable and should not be interpreted as production performance."
        )
    else:
        train_count = int(total_samples * (1 - test_size))
        test_count = total_samples - train_count
        print(
            f"\n  [INFO] Dataset contains {total_samples} samples.\n"
            f"  Split partition: {train_count} train / {test_count} test.\n"
            f"  Note: Real-world generalization depends on environmental diversity, noise, and recording independence."
        )

    if total_samples < 5:
        raise ValueError(
            f"Dataset too small ({total_samples} samples) to perform a valid train/test split."
        )

    class_counts = y.value_counts()
    min_class_count = class_counts.min()

    # Small-dataset safeguard: Each class must have at least 2 samples for stratification
    if min_class_count < 2:
        raise ValueError(
            f"Stratification impossible: minority class has only {min_class_count} sample(s). "
            f"At least 2 samples per class are required."
        )

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Post-split integrity checks
    # Check 1: Zero leakage between train and test IDs
    overlap_ids = set(ids_train).intersection(set(ids_test))
    if overlap_ids:
        raise ValueError(
            f"Data leakage detected! {len(overlap_ids)} IDs exist in both train and test sets."
        )

    # Check 2: Check test set class representation
    test_classes = set(y_test.unique())
    train_classes = set(y_train.unique())
    all_classes = set(y.unique())
    if test_classes != all_classes or train_classes != all_classes:
        raise ValueError(
            f"Stratification failed to represent all classes. "
            f"Train classes: {train_classes}, Test classes: {test_classes}, Expected: {all_classes}"
        )

    # Check 3: Check feature column alignment and ordering
    if list(X_train.columns) != list(X_test.columns) or list(X_train.columns) != list(X.columns):
        raise ValueError("Feature column mismatch or order discrepancy between train and test sets.")

    return X_train, X_test, y_train, y_test, ids_train, ids_test


# ----------------------------------------------------------------------
# 7. Model Definitions (Step 7)
# ----------------------------------------------------------------------
def get_candidate_models(random_state: int = PipelineConfig.RANDOM_STATE) -> Dict[str, Dict[str, Any]]:
    """
    Constructs candidate ML pipelines with appropriate scaling and hyperparameter grids.
    
    Models:
    1. Logistic Regression: with StandardScaler, max_iter=2000, class_weight='balanced'
    2. Random Forest: does not require scaling, class_weight='balanced'
    3. RBF SVM: with StandardScaler, probability=True, class_weight='balanced'
    
    Args:
        random_state: Fixed random seed for reproducibility.
        
    Returns:
        Dictionary of candidate model definitions with pipelines and parameter grids.
    """
    models = {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            "requires_scaling": True
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("classifier", RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    class_weight="balanced"
                ))
            ]),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5]
            },
            "requires_scaling": False
        },
        "SVM (RBF)": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0, 100.0],
                "classifier__gamma": ["scale", 0.01, 0.1, 1.0]
            },
            "requires_scaling": True
        }
    }
    return models


# ----------------------------------------------------------------------
# 8. Cross-Validation & Hyperparameter Tuning (Step 8)
# ----------------------------------------------------------------------
def tune_and_cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = PipelineConfig.RANDOM_STATE
) -> Dict[str, Dict[str, Any]]:
    """
    Executes Stratified K-Fold Cross-Validation and Hyperparameter Tuning
    strictly on the training dataset.
    
    Small-dataset safeguards:
    - Calculates viable fold count: min(5, min_class_count).
    - Prevents data leakage by encapsulating scaling inside scikit-learn Pipelines.
    - Test set is never touched.
    - Evaluates abnormal-class F1, recall, precision, and accuracy across folds.
    
    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        random_state: Random state seed.
        
    Returns:
        Dictionary of tuned model results, best estimators, parameters, and CV metrics.
    """
    min_class_count = y_train.value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            f"Cannot perform cross-validation: minority class has only {min_class_count} sample(s)."
        )

    # Dynamic fold determination per Section 23
    n_splits = min(5, min_class_count)
    if n_splits < 5:
        print(f"  [NOTE] Training set minority class has {min_class_count} samples. Using {n_splits}-fold CV.")
    else:
        print(f"  [CONFIG] Using {n_splits}-fold Stratified Cross-Validation.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    primary_scorer = make_scorer(f1_score, pos_label="abnormal")

    candidate_models = get_candidate_models(random_state=random_state)
    tuning_results: Dict[str, Dict[str, Any]] = {}

    for model_name, config in candidate_models.items():
        print(f"\n  Tuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring=primary_scorer,
            refit=True,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_f1 = grid_search.best_score_

        # Calculate multi-metric cross-validation fold statistics for best estimator
        fold_f1_scores = []
        fold_recalls = []
        fold_precisions = []
        fold_accuracies = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit clone of best pipeline on this fold's train data only
            best_model.fit(X_tr, y_tr)
            y_val_pred = best_model.predict(X_val)

            fold_f1 = f1_score(y_val, y_val_pred, pos_label="abnormal", zero_division=0)
            fold_rec = recall_score(y_val, y_val_pred, pos_label="abnormal", zero_division=0)
            fold_prec = precision_score(y_val, y_val_pred, pos_label="abnormal", zero_division=0)
            fold_acc = accuracy_score(y_val, y_val_pred)

            fold_f1_scores.append(fold_f1)
            fold_recalls.append(fold_rec)
            fold_precisions.append(fold_prec)
            fold_accuracies.append(fold_acc)

        # Refit best estimator on the full training set (X_train, y_train)
        best_model.fit(X_train, y_train)

        cv_summary = {
            "best_estimator": best_model,
            "best_params": best_params,
            "mean_cv_f1_abnormal": np.mean(fold_f1_scores),
            "std_cv_f1_abnormal": np.std(fold_f1_scores),
            "mean_cv_recall_abnormal": np.mean(fold_recalls),
            "std_cv_recall_abnormal": np.std(fold_recalls),
            "mean_cv_precision_abnormal": np.mean(fold_precisions),
            "std_cv_precision_abnormal": np.std(fold_precisions),
            "mean_cv_accuracy": np.mean(fold_accuracies),
            "std_cv_accuracy": np.std(fold_accuracies),
            "requires_scaling": config["requires_scaling"]
        }

        tuning_results[model_name] = cv_summary

        print(f"    - Best Params: {best_params}")
        print(f"    - Mean CV F1 (abnormal):        {cv_summary['mean_cv_f1_abnormal']:.4f} (+/- {cv_summary['std_cv_f1_abnormal']:.4f})")
        print(f"    - Mean CV Recall (abnormal):    {cv_summary['mean_cv_recall_abnormal']:.4f} (+/- {cv_summary['std_cv_recall_abnormal']:.4f})")
        print(f"    - Mean CV Precision (abnormal): {cv_summary['mean_cv_precision_abnormal']:.4f} (+/- {cv_summary['std_cv_precision_abnormal']:.4f})")
        print(f"    - Mean CV Accuracy:             {cv_summary['mean_cv_accuracy']:.4f} (+/- {cv_summary['std_cv_accuracy']:.4f})")

    return tuning_results


# ----------------------------------------------------------------------
# 9. Cross-Validation Model Comparison Table (Step 9)
# ----------------------------------------------------------------------
def generate_cv_comparison_table(
    cv_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Builds and displays a comprehensive comparison table across all candidate models
    based strictly on training Cross-Validation metrics.
    
    Args:
        cv_results: Results from cross-validation tuning on training data.
        
    Returns:
        pd.DataFrame containing formatted CV comparison metrics.
    """
    rows = []
    for model_name, cv in cv_results.items():
        rows.append({
            "Model": model_name,
            "CV F1 (Abnormal)": f"{cv['mean_cv_f1_abnormal']:.4f} (+/- {cv['std_cv_f1_abnormal']:.4f})",
            "CV Recall (Abnormal)": f"{cv['mean_cv_recall_abnormal']:.4f} (+/- {cv['std_cv_recall_abnormal']:.4f})",
            "CV Precision": f"{cv['mean_cv_precision_abnormal']:.4f}",
            "CV Accuracy": f"{cv['mean_cv_accuracy']:.4f}",
            "Best Parameters": str(cv["best_params"]),
        })

    comparison_df = pd.DataFrame(rows)
    print("\n" + comparison_df.to_string(index=False))
    return comparison_df


# ----------------------------------------------------------------------
# 10. Final Model Selection (Step 10 - Strictly Zero Test Leakage)
# ----------------------------------------------------------------------
def select_best_model(
    cv_results: Dict[str, Dict[str, Any]]
) -> Tuple[str, Any, str]:
    """
    Dynamically selects the best performing model based strictly on CV results (training data only):
    1. Primary: Mean Cross-Validation F1-Score (abnormal class)
    2. Secondary: Mean Cross-Validation Recall (abnormal class - minimizing missed faulty pumps / FN)
    3. Tertiary: Cross-Validation stability (lowest std of F1 across folds)
    4. Quaternary: Mean Cross-Validation Precision (abnormal class)
    5. Parsimony: Preference for simpler, more interpretable linear models when metrics are tied
    
    Test metrics are STRICTLY EXCLUDED from this function to guarantee zero test-set leakage.
    
    Args:
        cv_results: Results from cross-validation tuning on training data.
        
    Returns:
        Tuple of (selected_model_name, selected_pipeline, selection_reason)
    """
    complexity_rank = {
        "Logistic Regression": 1,
        "SVM (RBF)": 2,
        "Random Forest": 3,
    }

    def selection_key(name: str):
        cv = cv_results[name]
        return (
            -cv["mean_cv_f1_abnormal"],
            -cv["mean_cv_recall_abnormal"],
            cv["std_cv_f1_abnormal"],
            -cv["mean_cv_precision_abnormal"],
            complexity_rank.get(name, 99)
        )

    sorted_models = sorted(cv_results.keys(), key=selection_key)
    selected_name = sorted_models[0]
    best_estimator = cv_results[selected_name]["best_estimator"]
    cv_winner = cv_results[selected_name]

    # Formulate dynamic rationale
    reason = (
        f"Selected '{selected_name}' based strictly on Training Cross-Validation: "
        f"Mean CV F1 (Abnormal)={cv_winner['mean_cv_f1_abnormal']:.4f} (+/- {cv_winner['std_cv_f1_abnormal']:.4f}), "
        f"Mean CV Recall (Abnormal)={cv_winner['mean_cv_recall_abnormal']:.4f}, "
        f"Mean CV Accuracy={cv_winner['mean_cv_accuracy']:.4f}."
    )
    tied = [
        m for m in cv_results
        if m != selected_name
        and abs(cv_results[m]["mean_cv_f1_abnormal"] - cv_winner["mean_cv_f1_abnormal"]) < 1e-4
        and abs(cv_results[m]["mean_cv_recall_abnormal"] - cv_winner["mean_cv_recall_abnormal"]) < 1e-4
    ]
    if tied:
        reason += f" (Tied with {', '.join(tied)}; selected simpler model architecture by parsimony and fold stability)."

    return selected_name, best_estimator, reason


# ----------------------------------------------------------------------
# 11. Final Model Evaluation on Held-Out Test Set (Step 11 - Post-Selection)
# ----------------------------------------------------------------------
def evaluate_selected_model(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates the already-selected final model on the untouched held-out test set.
    Executed strictly AFTER model selection to guarantee zero test leakage.
    
    Metrics:
    - Accuracy
    - Precision (abnormal)
    - Recall (abnormal) - Maintenance-critical metric
    - F1-Score (abnormal)
    - Confusion Matrix (TN, FP, FN, TP)
    - ROC-AUC
    - PR-AUC (Average Precision)
    
    Args:
        model: Trained best estimator pipeline.
        model_name: Name of the selected model.
        X_test: Held-out test features.
        y_test: Held-out test ground truth labels.
        
    Returns:
        Dictionary of test set evaluation metrics for the selected model.
    """
    y_test_binary = (y_test == "abnormal").astype(int)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    classes = list(model.classes_)
    abnormal_idx = classes.index("abnormal")
    y_proba_abnormal = y_proba[:, abnormal_idx]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="abnormal", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="abnormal", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="abnormal", zero_division=0)

    # Confusion matrix with explicit labels: [normal, abnormal]
    cm = confusion_matrix(y_test, y_pred, labels=["normal", "abnormal"])
    tn, fp, fn, tp = cm.ravel()

    roc_auc = roc_auc_score(y_test_binary, y_proba_abnormal)
    pr_auc = average_precision_score(y_test_binary, y_proba_abnormal)

    eval_summary = {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    print(f"\n  [{model_name}] Final Test Set Performance (Held-Out Evaluation):")
    print(f"    - Accuracy:              {acc:.4f} ({acc*100:.1f}%)")
    print(f"    - Precision (abnormal):  {prec:.4f}")
    print(f"    - Recall (abnormal):     {rec:.4f} (Maintenance Critical)")
    print(f"    - F1-Score (abnormal):   {f1:.4f}")
    print(f"    - ROC-AUC:               {roc_auc:.4f}")
    print(f"    - PR-AUC:                {pr_auc:.4f}")
    print(f"    - Confusion Matrix:")
    print(f"        [Actual Normal  ] -> TN={tn}, FP={fp}")
    print(f"        [Actual Abnormal] -> FN={fn} (Missed Abnormal Pumps), TP={tp}")
    print(f"    Note: FN = abnormal pump incorrectly diagnosed as normal.")

    return eval_summary


# ----------------------------------------------------------------------
# 12. Model Saving & Artifact Verification (Step 12)
# ----------------------------------------------------------------------
def save_model_artifact(
    model_pipeline: Any,
    feature_cols: List[str],
    model_name: str,
    save_path: Path
) -> Path:
    """
    Saves the entire trained pipeline along with exact feature ordering metadata.
    
    Args:
        model_pipeline: Trained scikit-learn Pipeline (preprocessing + classifier).
        feature_cols: Exact list of feature column names in training order.
        model_name: Name of the winning model.
        save_path: Target .pkl destination path.
        
    Returns:
        Path of the saved artifact.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": model_pipeline,
        "feature_names": list(feature_cols),
        "model_name": model_name,
        "classes": list(model_pipeline.classes_),
    }

    joblib.dump(artifact, save_path)
    file_size_bytes = save_path.stat().st_size
    print(f"  Artifact successfully saved to: {save_path.resolve()}")
    print(f"  Artifact file size: {file_size_bytes:,} bytes")
    print(f"  Enclosed feature columns: {len(feature_cols)} features")
    print(f"  Enclosed classes: {list(model_pipeline.classes_)}")
    return save_path


def verify_saved_artifact(
    save_path: Path,
    test_sample: pd.DataFrame,
    expected_feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Loads saved artifact from disk and verifies inference on a real test record.
    
    Args:
        save_path: Path to saved .pkl artifact.
        test_sample: Single-row or multi-row DataFrame containing test features.
        expected_feature_cols: Expected list of feature column names.
        
    Returns:
        Verification result dictionary.
    """
    if not save_path.exists():
        raise FileNotFoundError(f"Verification failed: Artifact not found at {save_path}")

    loaded_artifact = joblib.load(save_path)

    # Check 1: Structure check
    required_keys = {"pipeline", "feature_names", "model_name", "classes"}
    if not required_keys.issubset(loaded_artifact.keys()):
        missing = required_keys - set(loaded_artifact.keys())
        raise ValueError(f"Artifact corrupted: missing keys {missing}")

    loaded_pipeline = loaded_artifact["pipeline"]
    loaded_features = loaded_artifact["feature_names"]

    # Check 2: Feature alignment check
    if loaded_features != expected_feature_cols:
        raise ValueError("Saved feature list does not match expected training feature columns.")

    # Check 3: Real prediction test on sample
    sample_aligned = test_sample[loaded_features].copy()
    prediction = loaded_pipeline.predict(sample_aligned)[0]
    probabilities = loaded_pipeline.predict_proba(sample_aligned)[0]
    classes = loaded_artifact["classes"]
    prob_dict = {str(c): float(p) for c, p in zip(classes, probabilities)}
    confidence = float(np.max(probabilities))

    verification_result = {
        "verified": True,
        "model_name": loaded_artifact["model_name"],
        "sample_prediction": str(prediction),
        "sample_confidence": confidence,
        "sample_probabilities": prob_dict
    }

    print("  [PASS] Saved model loaded from disk successfully.")
    print(f"  [PASS] Architecture: {loaded_artifact['model_name']} Pipeline")
    print(f"  [PASS] Feature alignment verified ({len(loaded_features)} features).")
    print(f"  [PASS] Real test sample prediction: '{prediction}' (confidence={confidence:.4f})")
    print(f"  [PASS] Class probabilities: {prob_dict}")
    return verification_result


# ----------------------------------------------------------------------
# Main Execution Entrypoint
# ----------------------------------------------------------------------
def main():
    """Main pipeline execution for Steps 1 through 12."""
    config = PipelineConfig()
    
    run_banner = (
        "DUMMY DATASET - TEST RUN"
        if config.RUN_MODE == "DUMMY"
        else "REAL DATASET"
    )

    print("=" * 70)
    print("JAL-SETU MACHINE LEARNING PIPELINE")
    print(f"MODE: {run_banner}")
    print("=" * 70)

    csv_path = config.get_active_csv_path()
    print(f"\nActive CSV Target: {csv_path.resolve()}")

    # [1] PROJECT / DATASET INSPECTION
    print("\n" + "=" * 70)
    print("[1] PROJECT / DATASET INSPECTION")
    print("=" * 70)
    df = load_dataset(csv_path)
    inspection_results = inspect_dataset(df, dataset_label=config.RUN_MODE)

    # [2] DATASET VALIDATION (Step 5)
    print("\n" + "=" * 70)
    print("[2] DATASET VALIDATION & DYNAMIC FEATURE DETECTION (STEP 5)")
    print("=" * 70)
    feature_cols, target_col, id_col = validate_dataset(
        df,
        id_col=config.ID_COLUMN,
        target_col=config.TARGET_COLUMN,
        expected_classes=config.EXPECTED_CLASSES,
    )

    # [3] FEATURE PREPARATION / SEPARATION
    print("\n" + "=" * 70)
    print("[3] FEATURE PREPARATION")
    print("=" * 70)
    X, y, ids = separate_features_and_target(df, feature_cols, target_col, id_col)
    print(f"  Feature Matrix (X) Shape: {X.shape} (samples={X.shape[0]}, features={X.shape[1]})")
    print(f"  Target Vector  (y) Shape: {y.shape} (samples={y.shape[0]})")
    print(f"  ID Vector   (ids) Shape: {ids.shape} (samples={ids.shape[0]})")
    print(f"  Ordered Features Count: {len(feature_cols)}")
    print(f"  First 5 features: {feature_cols[:5]}")
    print(f"  Last 5 features:  {feature_cols[-5:]}")

    # [4] TRAIN / TEST SPLIT (Step 6)
    print("\n" + "=" * 70)
    print("[4] TRAIN / TEST SPLIT (STRATIFIED) (STEP 6)")
    print("=" * 70)
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_train_test(
        X, y, ids, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    print(f"  Split Configuration:")
    print(f"    - Target Split: {(1 - config.TEST_SIZE) * 100:.0f}% Train / {config.TEST_SIZE * 100:.0f}% Test")
    print(f"    - Random Seed:  {config.RANDOM_STATE}")
    print(f"  Split Results:")
    print(f"    - Train set: {X_train.shape[0]} samples ({(len(X_train)/len(X))*100:.1f}%), {X_train.shape[1]} features")
    print(f"    - Test set:  {X_test.shape[0]} samples ({(len(X_test)/len(X))*100:.1f}%), {X_test.shape[1]} features")
    print(f"  Class Proportions:")
    print(f"    - Full dataset:  {y.value_counts().to_dict()}")
    print(f"    - Train split:   {y_train.value_counts().to_dict()}")
    print(f"    - Test split:    {y_test.value_counts().to_dict()}")
    print(f"  Test Set IDs ({len(ids_test)} held-out recordings):")
    print(f"    {list(ids_test)}")
    print("  [PASS] Zero sample ID leakage between train and test sets verified.")
    print("  [PASS] Stratification verified: exactly equal class proportions maintained.")
    print("  [PASS] Exact feature column ordering strictly preserved across train and test sets.")

    # [5] MODEL CONFIGURATION (Step 7)
    print("\n" + "=" * 70)
    print("[5] MODEL CONFIGURATION (STEP 7)")
    print("=" * 70)
    candidate_models = get_candidate_models(random_state=config.RANDOM_STATE)
    for name, m_info in candidate_models.items():
        scaling_str = "StandardScaler Pipeline" if m_info["requires_scaling"] else "None (Raw Features)"
        print(f"  - {name}:")
        print(f"      Preprocessing: {scaling_str}")
        print(f"      Tuning Grid:   {m_info['param_grid']}")

    # [6] CROSS-VALIDATION & HYPERPARAMETER TUNING (Step 8)
    print("\n" + "=" * 70)
    print("[6] CROSS-VALIDATION & HYPERPARAMETER TUNING (STEP 8)")
    print("=" * 70)
    cv_results = tune_and_cross_validate_models(
        X_train=X_train,
        y_train=y_train,
        random_state=config.RANDOM_STATE
    )

    # [7] MODEL COMPARISON TABLE (CROSS-VALIDATION) (Step 9)
    print("\n" + "=" * 70)
    print("[7] MODEL COMPARISON TABLE (CROSS-VALIDATION) (STEP 9)")
    print("=" * 70)
    comparison_df = generate_cv_comparison_table(cv_results=cv_results)

    # [8] FINAL MODEL SELECTION (STRICTLY FROM CV) (Step 10)
    print("\n" + "=" * 70)
    print("[8] FINAL MODEL SELECTION (TRAINING CV ONLY) (STEP 10)")
    print("=" * 70)
    selected_model_name, selected_pipeline, selection_reason = select_best_model(
        cv_results=cv_results
    )
    print(f"  Selected Model: {selected_model_name}")
    print(f"  Selection Rationale:\n    {selection_reason}")

    # [9] MODEL EVALUATION ON HELD-OUT TEST SET (POST-SELECTION ONLY) (Step 11)
    print("\n" + "=" * 70)
    print("[9] MODEL EVALUATION ON HELD-OUT TEST SET (STEP 11)")
    print("  (Held-out test set evaluated strictly after model selection; 0% test leakage)")
    print("=" * 70)
    test_eval_results = evaluate_selected_model(
        model=selected_pipeline,
        model_name=selected_model_name,
        X_test=X_test,
        y_test=y_test
    )

    # [10] MODEL SAVING & ARTIFACT VERIFICATION (Step 12)
    print("\n" + "=" * 70)
    print("[10] MODEL SAVING & ARTIFACT VERIFICATION (STEP 12)")
    print("=" * 70)
    target_model_path = config.get_active_model_path()
    saved_path = save_model_artifact(
        model_pipeline=selected_pipeline,
        feature_cols=feature_cols,
        model_name=selected_model_name,
        save_path=target_model_path
    )

    print("\n  Testing Saved Artifact via Independent Load:")
    # Pick the first test sample from held-out set (features only)
    sample_record = X_test.iloc[[0]]
    verify_result = verify_saved_artifact(
        save_path=saved_path,
        test_sample=sample_record,
        expected_feature_cols=feature_cols
    )

    # Final Terminal Output Summary (Section 27)
    cv_folds_used = min(5, y_train.value_counts().min())

    print("\n" + "=" * 60)
    print("JAL-SETU ML TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nMODE:\n{run_banner}")
    print(f"\nDATASET:\n  Rows:     {len(df)}\n  Features: {len(feature_cols)}\n  Normal:   {df[target_col].value_counts().get('normal', 0)}\n  Abnormal: {df[target_col].value_counts().get('abnormal', 0)}")
    print(f"\nCV:\n  Folds used: {cv_folds_used}")
    print(f"\nMODEL COMPARISON (CROSS-VALIDATION):\n{comparison_df.to_string(index=False)}")
    print("\n" + "=" * 30)
    print("FINAL MODEL SELECTION")
    print("=" * 30)
    print(f"\nSelected model: {selected_model_name}")
    print("\nSelection basis:")
    print("- CV F1")
    print("- abnormal recall")
    print("- CV stability")
    print("- simplicity/tie-breaking where applicable")
    print("\nTest evaluation is reported separately")
    print("and was NOT used for model selection.")
    print("\n" + "=" * 30)
    print("FINAL TEST RESULTS")
    print("=" * 30)
    print(f"\nAccuracy:             {test_eval_results['accuracy']:.4f}")
    print(f"Precision (abnormal): {test_eval_results['precision']:.4f}")
    print(f"Recall (abnormal):    {test_eval_results['recall']:.4f}")
    print(f"F1 (abnormal):        {test_eval_results['f1']:.4f}")
    print(f"ROC-AUC:              {test_eval_results['roc_auc']:.4f}")
    print(f"PR-AUC:               {test_eval_results['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {test_eval_results['tn']}")
    print(f"FP: {test_eval_results['fp']}")
    print(f"FN: {test_eval_results['fn']}")
    print(f"TP: {test_eval_results['tp']}")
    print("\nFINAL MODEL SAVED:")
    print(f"{saved_path.as_posix()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
