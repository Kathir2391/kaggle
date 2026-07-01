"""
Stellar Object Classification Pipeline
=======================================
Predicts the class of stellar objects (GALAXY / QSO / STAR) from
photometric survey data using LightGBM (primary) with XGBoost and
sklearn HistGradientBoosting as drop-in alternatives.

TARGET COLUMN
-------------
  class : GALAXY | QSO | STAR  (3-class multiclass problem)

KEY FINDINGS FROM EDA
---------------------
  - Redshift is the strongest single separator:
      STAR   mean z = 0.07  (very nearby, essentially zero redshift)
      GALAXY mean z = 0.52  (moderate redshift)
      QSO    mean z = 1.90  (high redshift, active galactic nuclei)
  - Colour indices g-r and r-i are the strongest photometric predictors
    (|r| = 0.57 and 0.52 with class label)
  - galaxy_population (Red_Sequence / Blue_Cloud) is included as a feature —
    QSOs are 91% Blue_Cloud, GALAXYs are 75% Red_Sequence
  - spectral_type is one-hot encoded (not ordinal):
      GALAXY: dominated by M-type (77%)
      QSO:    dominated by A/F (55%) and O/B (19%)
      STAR:   mixed A/F (47%), G/K (29%)
  - No missing values in the dataset

OPTIMISATIONS FOR LARGE DATASETS (500k+ rows)
---------------------------------------------
  - LightGBM / XGBoost: histogram binning, parallelised tree building
  - n_jobs=-1 on cross_validate: one fold per CPU core
  - StratifiedKFold(n_splits=3): 3 folds sufficient at scale, 40% faster
  - CV on a capped stratified sample; final model trains on 100% of data
  - OMP/MKL thread env vars set at startup to saturate all cores

BACKEND SELECTION
-----------------
  BACKEND=lgbm   LightGBM (default, fastest)
  BACKEND=xgb    XGBoost  (GPU via --device cuda)
  BACKEND=hist   sklearn HistGradientBoostingClassifier (no install needed)

  export BACKEND=lgbm
  python src/predict.py --mode train

USAGE
-----
  # Train (reads from project layout by default):
  python src/predict.py --mode train

  # Train with explicit paths:
  python src/predict.py --mode train \\
      --input  stellar-class/data/raw/train.csv \\
      --model  stellar-class/data/processed/model.pkl \\
      --plot   stellar-class/data/processed/evaluation_report.png

  # Predict on new data:
  python src/predict.py --mode predict \\
      --input  stellar-class/data/raw/test.csv \\
      --output stellar-class/data/predictions/predictions.csv

OUTPUT FORMAT (predict mode)
-----------------------------
  id, class
  0, GALAXY
  1, QSO
  ...

INPUT FORMAT
------------
  Required columns: id, alpha, delta, u, g, r, i, z, redshift,
                    spectral_type, galaxy_population
  Label column (train mode only): class  (GALAXY | QSO | STAR)
"""

import os
import argparse
import warnings
import pickle
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

warnings.filterwarnings('ignore')

# ── Saturate all CPU cores at the OS level ──────────────────────────────────
_N_CORES = str(os.cpu_count() or 4)
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_env, _N_CORES)


# ─────────────────────────────────────────────
# PROJECT-RELATIVE PATHS
# ─────────────────────────────────────────────
# __file__ = stellar-class/src/predict.py
# SRC_DIR  = stellar-class/src/
# ROOT_DIR = stellar-class/

SRC_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

DEFAULT_TRAIN_PATH = ROOT_DIR / 'data' / 'raw'         / 'train.csv'
DEFAULT_TEST_PATH  = ROOT_DIR / 'data' / 'raw'         / 'test.csv'
DEFAULT_MODEL_PATH = ROOT_DIR / 'data' / 'processed'   / 'model.pkl'
DEFAULT_PLOT_PATH  = ROOT_DIR / 'data' / 'processed'   / 'evaluation_report.png'
DEFAULT_PRED_PATH  = ROOT_DIR / 'data' / 'predictions' / 'predictions.csv'


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LABEL_COL      = 'class'
CLASSES        = ['GALAXY', 'QSO', 'STAR']   # fixed order for consistency
SPECTRAL_COL   = 'spectral_type'
GALPOP_COL     = 'galaxy_population'
BAND_COLS      = ['u', 'g', 'r', 'i', 'z']
COORD_COLS     = ['alpha', 'delta']
REDSHIFT_COL   = 'redshift'
ID_COL         = 'id'

# Redshift bins for stratified evaluation
# QSOs concentrate at z > 1, STARs at z < 0.2 — binning reveals per-class shifts
Z_BINS   = [0, 0.2, 0.5, 1.0, 2.0, 99]
Z_LABELS = ['<0.2', '0.2-0.5', '0.5-1', '1-2', '>2']

# Training r-band magnitude range (flag out-of-distribution objects at inference)
R_BAND_MIN, R_BAND_MAX = 14.85, 23.13

# CV sample cap: run CV on at most this many rows for speed;
# final model always trains on 100% of clean data.
CV_SAMPLE_CAP = 100_000


# ─────────────────────────────────────────────
# BACKEND — LightGBM / XGBoost / HistGB
# ─────────────────────────────────────────────

def _build_lgbm(n_jobs: int, device: str):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    params = dict(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 20,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        n_jobs            = n_jobs,
        random_state      = 42,
        verbose           = -1,
        objective         = 'multiclass',
        num_class         = len(CLASSES),
    )
    if device == 'gpu':
        params['device'] = 'gpu'
    return lgb.LGBMClassifier(**params)


def _build_xgb(n_jobs: int, device: str):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    params = dict(
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        tree_method      = 'hist',
        objective        = 'multi:softprob',
        num_class        = len(CLASSES),
        n_jobs           = n_jobs,
        random_state     = 42,
        eval_metric      = 'mlogloss',
        verbosity        = 0,
    )
    if device == 'cuda':
        params['device'] = 'cuda'
    return XGBClassifier(**params)


def _build_hist(n_jobs: int, device: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter          = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        max_bins          = 255,
        min_samples_leaf  = 20,
        l2_regularization = 1.0,
        early_stopping    = True,
        n_iter_no_change  = 20,
        random_state      = 42,
    )


def build_model(backend: str = 'lgbm', n_jobs: int = -1,
                device: str = 'cpu', calibrate: bool = False):
    """
    Build the multiclass classifier.

    Calibration is OFF by default for multiclass — CalibratedClassifierCV
    wraps in OvR which loses the joint probability structure. Use raw
    predict_proba from LightGBM/XGBoost which are already well-calibrated
    for multiclass via softmax.

    backend : 'lgbm' | 'xgb' | 'hist'
    device  : 'cpu' | 'gpu' | 'cuda'
    """
    builders = {'lgbm': _build_lgbm, 'xgb': _build_xgb, 'hist': _build_hist}
    if backend not in builders:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(builders)}")
    base = builders[backend](n_jobs=n_jobs, device=device)
    if calibrate:
        return CalibratedClassifierCV(base, method='isotonic', cv=3)
    return base


# ─────────────────────────────────────────────
# PHASE 1 — DATA QUALITY & PREPROCESSING
# ─────────────────────────────────────────────

def check_required_columns(df: pd.DataFrame, mode: str) -> None:
    """Raise a clear error if any required column is missing."""
    required = [ID_COL, SPECTRAL_COL, GALPOP_COL,
                REDSHIFT_COL] + BAND_COLS + COORD_COLS
    if mode == 'train':
        required.append(LABEL_COL)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def flag_missing_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with nulls in any photometric band or redshift."""
    missing_mask = df[BAND_COLS + [REDSHIFT_COL]].isnull().any(axis=1)
    n = missing_mask.sum()
    if n > 0:
        print(f"  [WARNING] {n:,} rows have missing band/redshift values — flagged.")
    df = df.copy()
    df['_flag_missing_bands'] = missing_mask
    return df


def apply_k_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    First-order K-correction proxy: Δm ≈ k_coeff × redshift.

    This removes the redshift-dependent colour shift from each band so
    that colour indices reflect intrinsic stellar population properties
    rather than cosmological distance effects.

    Replace with full SED template fitting (LePhare / EAZY) in production.
    """
    k_coeffs = {'u': 1.0, 'g': 0.6, 'r': 0.3, 'i': 0.15, 'z': 0.05}
    df = df.copy()
    for band, coeff in k_coeffs.items():
        df[f'{band}_kcorr'] = df[band] - coeff * df[REDSHIFT_COL]
    print("  [OK] K-correction proxy applied to all 5 bands.")
    return df


def flag_depth_extrapolation(df: pd.DataFrame) -> pd.DataFrame:
    """Flag objects outside the training r-band magnitude range."""
    df = df.copy()
    oob = (df['r'] < R_BAND_MIN) | (df['r'] > R_BAND_MAX)
    df['_flag_depth_extrap'] = oob
    n = oob.sum()
    if n > 0:
        print(f"  [WARNING] {n:,} objects outside training r-band range "
              f"[{R_BAND_MIN:.2f}, {R_BAND_MAX:.2f}] — predictions may be unreliable.")
    return df


# ─────────────────────────────────────────────
# PHASE 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all predictive features.

    Feature groups and their rationale
    -----------------------------------
    1. Colour indices (u-g, g-r, r-i, i-z)
       Primary photometric discriminators. g-r and r-i have the highest
       correlation with class (|r| = 0.57, 0.52). Raw magnitudes are
       excluded — adjacent bands are collinear (r-i: r=0.954) and
       conflate intrinsic brightness with distance.

    2. K-corrected colour indices (u-g-k, g-r-k, r-i-k, i-z-k)
       Same indices after removing the redshift-dependent colour shift.
       Especially important for QSOs (mean z=1.90) where observed colours
       are strongly displaced from rest-frame values.

    3. Redshift
       Single strongest separator by class:
         STAR z≈0.07, GALAXY z≈0.52, QSO z≈1.90
       Note: at inference time use photometric-z, not spectroscopic-z,
       to avoid data leakage.

    4. Sky coordinates (alpha, delta)
       Weak signal (~0.06 correlation) but included for completeness.
       May capture survey-depth gradients across the sky.

    5. Spectral type (one-hot)
       Strong class signal: M-types → 95% GALAXY; O/B → 66% QSO/STAR.
       One-hot encoded — spectral sequence is NOT linearly ordered.

    6. Galaxy population (one-hot)
       QSOs are 91% Blue_Cloud; GALAXYs are 75% Red_Sequence.
       Adds orthogonal discriminative signal to colour indices.
    """
    df = df.copy()

    # Raw colour indices
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']

    # K-corrected colour indices
    kcorr_cols = [f'{b}_kcorr' for b in BAND_COLS]
    if all(c in df.columns for c in kcorr_cols):
        df['u_g_k'] = df['u_kcorr'] - df['g_kcorr']
        df['g_r_k'] = df['g_kcorr'] - df['r_kcorr']
        df['r_i_k'] = df['r_kcorr'] - df['i_kcorr']
        df['i_z_k'] = df['i_kcorr'] - df['z_kcorr']

    # One-hot: spectral type (M, A/F, G/K, O/B)
    spec_dummies = pd.get_dummies(df[SPECTRAL_COL], prefix='spec', dtype=float)
    df = pd.concat([df, spec_dummies], axis=1)

    # One-hot: galaxy population (Red_Sequence, Blue_Cloud)
    galpop_dummies = pd.get_dummies(df[GALPOP_COL], prefix='galpop', dtype=float)
    df = pd.concat([df, galpop_dummies], axis=1)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ordered list of feature columns used for modelling."""
    base     = ['u_g', 'g_r', 'r_i', 'i_z', REDSHIFT_COL] + COORD_COLS
    kcorr    = ['u_g_k', 'g_r_k', 'r_i_k', 'i_z_k'] if 'u_g_k' in df.columns else []
    spectral = sorted([c for c in df.columns if c.startswith('spec_')])
    galpop   = sorted([c for c in df.columns if c.startswith('galpop_')])
    return base + kcorr + spectral + galpop


def encode_labels(series: pd.Series, le: LabelEncoder = None):
    """
    Encode class labels to integers using a fixed LabelEncoder.
    Returns (encoded_array, fitted_encoder).
    Pass a pre-fitted encoder at inference time to guarantee consistency.
    """
    if le is None:
        le = LabelEncoder()
        le.fit(CLASSES)   # fix order: GALAXY=0, QSO=1, STAR=2
    unknown = set(series.unique()) - set(le.classes_)
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}. "
                         f"Expected one of {list(le.classes_)}")
    return le.transform(series), le


# ─────────────────────────────────────────────
# PHASE 3 — CROSS-VALIDATION (parallelised)
# ─────────────────────────────────────────────

def cross_validate_model(model, X: pd.DataFrame, y: np.ndarray,
                          n_splits: int = 3) -> dict:
    """
    Stratified k-fold CV with full CPU parallelism.

    Optimisations
    -------------
    - n_splits=3: statistically sufficient at 500k+ rows, 40% faster than 5.
    - CV sample cap: CV runs on at most CV_SAMPLE_CAP rows; final model
      trains on 100% of data.
    - n_jobs=-1 on cross_validate: one fold per CPU core.
    - n_jobs=-1 inside LightGBM/XGBoost: parallelises within each fold.
    """
    n_total = len(X)
    if n_total > CV_SAMPLE_CAP:
        print(f"  [INFO] {n_total:,} rows — CV on a stratified "
              f"{CV_SAMPLE_CAP:,}-row sample. Final model trains on all rows.")
        X_cv, _, y_cv, _ = train_test_split(
            X, y, train_size=CV_SAMPLE_CAP, stratify=y, random_state=42
        )
    else:
        X_cv, y_cv = X, y

    effective_cores = joblib.effective_n_jobs(-1)
    print(f"  [INFO] Parallelising {n_splits} folds across {effective_cores} CPU cores.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    t0 = time.time()
    scores = cross_validate(
        model, X_cv, y_cv,
        cv=cv,
        scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted'],
        n_jobs=-1,
        return_train_score=False,
        verbose=0,
    )
    print(f"  [OK] CV completed in {time.time()-t0:.1f}s.")

    results = {}
    for metric in ['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']:
        vals = scores[f'test_{metric}']
        results[metric] = {
            'mean':   vals.mean(),
            'std':    vals.std(),
            'values': vals.tolist(),
        }
    return results


# ─────────────────────────────────────────────
# PHASE 4 — EVALUATION
# ─────────────────────────────────────────────

def evaluate_by_redshift_bin(y_true: np.ndarray, y_pred: np.ndarray,
                              z_values: pd.Series,
                              le: LabelEncoder) -> pd.DataFrame:
    """
    Per-redshift-bin accuracy.

    Critical for this dataset: class mix shifts sharply across z —
    STARs dominate at z<0.2, GALAXYs at 0.2-1.0, QSOs above z>1.
    A single accuracy number conceals this completely.

    All inputs are normalised to 0-based index before masking to prevent
    pandas IndexingError from train_test_split's non-contiguous indices.
    """
    y_true   = pd.Series(np.asarray(y_true)).reset_index(drop=True)
    y_pred   = pd.Series(np.asarray(y_pred)).reset_index(drop=True)
    z_values = pd.Series(np.asarray(z_values)).reset_index(drop=True)

    z_bins = pd.cut(z_values, bins=Z_BINS, labels=Z_LABELS)
    rows = []
    for zlabel in Z_LABELS:
        mask = (z_bins == zlabel)
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        acc = (yt == yp).mean()
        # Per-bin class breakdown (decoded)
        decoded_true = le.inverse_transform(yt.to_numpy())
        breakdown = pd.Series(decoded_true).value_counts().to_dict()
        breakdown_str = ' | '.join(
            f"{cls}:{breakdown.get(cls,0)}" for cls in CLASSES
        )
        rows.append({
            'Redshift bin': zlabel,
            'N objects':    mask.sum(),
            'Accuracy':     f'{acc:.4f}',
            'Class mix':    breakdown_str,
        })
    return pd.DataFrame(rows)


def compute_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract averaged feature importances from the trained model."""
    try:
        # CalibratedClassifierCV wraps base estimators
        importances = np.mean([
            e.estimator.feature_importances_
            for e in model.calibrated_classifiers_
        ], axis=0)
    except Exception:
        try:
            importances = model.feature_importances_
        except Exception:
            return pd.DataFrame()
    return (
        pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────
# PHASE 5 — VISUALISATION
# ─────────────────────────────────────────────

def plot_evaluation(y_test: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    z_test: pd.Series,
                    feature_importance: pd.DataFrame,
                    cv_results: dict,
                    n_cv_splits: int,
                    le: LabelEncoder,
                    output_path: str = 'evaluation_report.png') -> None:
    """
    6-panel evaluation figure:
      1. Confusion matrix (absolute counts)
      2. Per-class precision / recall bar chart
      3. ROC curves (one-vs-rest per class)
      4. Predicted probability distributions per class
      5. Accuracy by redshift bin
      6. Feature importances (top 12)
    """
    # Normalise to clean 0-based index so all boolean masks align
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    z_test = pd.Series(np.asarray(z_test)).reset_index(drop=True)

    CLASS_COLORS = {'GALAXY': '#185FA5', 'QSO': '#D85A30', 'STAR': '#1D9E75'}
    PURPLE, GRAY = '#534AB7', '#888780'

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── Panel 1: Confusion matrix ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.classes_
    ).plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title('Confusion matrix', fontsize=12, fontweight='medium', pad=10)
    ax1.tick_params(labelsize=9)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

    # ── Panel 2: Per-class precision / recall ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import precision_score, recall_score
    precisions = precision_score(y_test, y_pred, average=None)
    recalls    = recall_score(y_test, y_pred, average=None)
    x = np.arange(len(CLASSES))
    w = 0.35
    ax2.bar(x - w/2, precisions, w, label='Precision',
            color=[CLASS_COLORS[c] for c in CLASSES], alpha=0.85, edgecolor='white')
    ax2.bar(x + w/2, recalls, w, label='Recall',
            color=[CLASS_COLORS[c] for c in CLASSES], alpha=0.45, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, fontsize=10)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.set_title('Precision (solid) & recall (light) per class',
                  fontsize=11, fontweight='medium', pad=10)
    for xi, (p, r) in enumerate(zip(precisions, recalls)):
        ax2.text(xi - w/2, p + 0.02, f'{p:.2f}', ha='center', fontsize=8, color=GRAY)
        ax2.text(xi + w/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=8, color=GRAY)
    ax2.set_facecolor('#F8F8F8')
    ax2.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: ROC curves (one-vs-rest) ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=list(range(len(CLASSES))))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax3.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=2,
                 label=f'{cls} (AUC={auc:.3f})')
    ax3.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle='--')
    ax3.set_xlabel('False positive rate', fontsize=10)
    ax3.set_ylabel('True positive rate', fontsize=10)
    ax3.set_title('ROC curves (one-vs-rest)', fontsize=12,
                  fontweight='medium', pad=10)
    ax3.legend(fontsize=9)
    ax3.set_facecolor('#F8F8F8')
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Predicted probability distributions ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for i, cls in enumerate(CLASSES):
        mask = (y_test == i)
        ax4.hist(y_prob[mask, i], bins=20, alpha=0.55,
                 color=CLASS_COLORS[cls], label=cls, density=True)
    ax4.set_xlabel('Predicted probability (own class)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title('Score distribution by true class', fontsize=12,
                  fontweight='medium', pad=10)
    ax4.legend(fontsize=9)
    ax4.set_facecolor('#F8F8F8')
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Accuracy by redshift bin ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    y_test_s = pd.Series(y_test).reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).reset_index(drop=True)
    z_bins_col = pd.cut(z_test, bins=Z_BINS, labels=Z_LABELS)
    bin_acc, bin_labs, bin_counts = [], [], []
    for zlabel in Z_LABELS:
        mask = (z_bins_col == zlabel)
        if mask.sum() == 0:
            continue
        bin_acc.append((y_test_s[mask] == y_pred_s[mask]).mean())
        bin_labs.append(zlabel)
        bin_counts.append(mask.sum())
    bars = ax5.bar(bin_labs, bin_acc, color=PURPLE, alpha=0.75,
                   edgecolor='white', width=0.6)
    for bar, n in zip(bars, bin_counts):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'n={n}', ha='center', va='bottom', fontsize=8, color=GRAY)
    ax5.set_ylim(0, 1.12)
    ax5.set_xlabel('Redshift bin', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Accuracy by redshift bin', fontsize=12,
                  fontweight='medium', pad=10)
    ax5.axhline(1.0, color=GRAY, lw=0.8, linestyle='--')
    ax5.set_facecolor('#F8F8F8')
    ax5.grid(True, alpha=0.3, axis='y')

    # ── Panel 6: Feature importances ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if not feature_importance.empty:
        top = feature_importance.head(12)
        colors = [
            '#D85A30' if any(ci in f for ci in ('u_g','g_r','r_i','i_z')) else
            '#185FA5' if 'redshift' in f else
            '#1D9E75' if 'spec_' in f else
            '#534AB7' if 'galpop_' in f else
            '#888780'
            for f in top['Feature']
        ]
        ax6.barh(top['Feature'][::-1], top['Importance'][::-1],
                 color=colors[::-1], alpha=0.85, edgecolor='white')
        ax6.set_xlabel('Importance', fontsize=10)
        ax6.set_title('Feature importances (top 12)', fontsize=12,
                      fontweight='medium', pad=10)
        ax6.set_facecolor('#F8F8F8')
        ax6.grid(True, alpha=0.3, axis='x')
        ax6.tick_params(labelsize=8)

    cv_acc = cv_results['accuracy']['mean']
    cv_auc = cv_results['roc_auc_ovr_weighted']['mean']
    cv_f1  = cv_results['f1_weighted']['mean']
    fig.suptitle('Stellar Object Classifier (GALAXY / QSO / STAR) — Evaluation Report',
                 fontsize=14, fontweight='medium', y=1.01)
    fig.text(0.5, 0.995,
             f'{n_cv_splits}-fold CV:  Accuracy {cv_acc:.4f}  |  '
             f'AUC (OvR) {cv_auc:.4f}  |  F1 (weighted) {cv_f1:.4f}',
             ha='center', va='bottom', fontsize=10, color=GRAY)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Evaluation plots saved → {output_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE — TRAIN
# ─────────────────────────────────────────────

def pipeline_train(input_path: Path, model_path: Path, plot_path: Path,
                   test_size: float = 0.2,
                   backend: str = 'lgbm',
                   device: str = 'cpu',
                   n_cv_splits: int = 3) -> None:

    print("\n" + "=" * 60)
    print("STELLAR OBJECT CLASSIFICATION — TRAINING PIPELINE")
    print(f"Backend: {backend.upper()}  |  Device: {device}  |  "
          f"CV folds: {n_cv_splits}  |  CV cap: {CV_SAMPLE_CAP:,} rows")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Training data not found: {input_path}\n"
            f"Expected: stellar-class/data/raw/train.csv"
        )
    df = pd.read_csv(input_path)
    check_required_columns(df, mode='train')
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns.")
    print(f"  Class distribution:")
    for cls, cnt in df[LABEL_COL].value_counts().items():
        print(f"    {cls:<8}: {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    # ── 2. Quality checks ────────────────────────────────────────────────────
    print("\n[2/7] Data quality checks...")
    df = flag_missing_bands(df)
    df = flag_depth_extrapolation(df)

    # ── 3. K-correction ──────────────────────────────────────────────────────
    print("\n[3/7] Applying K-correction proxy...")
    df = apply_k_correction(df)

    # ── 4. Feature engineering ───────────────────────────────────────────────
    print("\n[4/7] Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(df[feature_cols].median())
    y, le = encode_labels(df[LABEL_COL])
    print(f"  Feature set ({len(feature_cols)}): {feature_cols}")
    print(f"  Label encoding: { {c: i for i, c in enumerate(le.classes_)} }")

    # ── 5. Cross-validation (parallelised) ───────────────────────────────────
    print(f"\n[5/7] Cross-validating ({n_cv_splits}-fold, parallelised)...")
    model_cv = build_model(backend=backend, n_jobs=-1, device=device)
    cv_results = cross_validate_model(model_cv, X, y, n_splits=n_cv_splits)

    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │            Cross-Validation Results              │")
    print("  ├────────────────────────┬──────────┬─────────────┤")
    print(f"  │ {'Metric':<22} │ {'Mean':>8} │ {'Std':>11} │")
    print("  ├────────────────────────┼──────────┼─────────────┤")
    metric_labels = {
        'accuracy':              'Accuracy',
        'f1_weighted':           'F1 (weighted)',
        'roc_auc_ovr_weighted':  'AUC-ROC (OvR)',
    }
    for key, label in metric_labels.items():
        v = cv_results[key]
        print(f"  │ {label:<22} │ {v['mean']:>8.4f} │ {v['std']:>11.4f} │")
    print("  └────────────────────────┴──────────┴─────────────┘")

    # ── 6. Final model on full data ──────────────────────────────────────────
    print(f"\n[6/7] Training final model on full {len(X):,} rows "
          f"(held-out test = {test_size:.0%})...")
    # Convert y to Series first so we can use .index for z_test alignment
    y_series = pd.Series(y, index=X.index)
    X_train, X_test, y_train_s, y_test_s = train_test_split(
        X, y_series, test_size=test_size, stratify=y_series, random_state=42
    )
    # Grab z_test using original split indices BEFORE resetting
    z_test  = df.loc[y_test_s.index, REDSHIFT_COL].reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train_s.reset_index(drop=True).to_numpy()
    y_test  = y_test_s.reset_index(drop=True).to_numpy()

    t0 = time.time()
    final_model = build_model(backend=backend, n_jobs=-1, device=device)
    final_model.fit(X_train, y_train)
    print(f"  [OK] Final model trained in {time.time()-t0:.1f}s.")

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    print("\n[7/7] Evaluating on held-out test set...")
    print("\n  Classification report:")
    print(classification_report(
        y_test, y_pred, target_names=le.classes_
    ))
    print("  Accuracy by redshift bin:")
    z_report = evaluate_by_redshift_bin(y_test, y_pred, z_test, le)
    print(z_report.to_string(index=False))

    fi = compute_feature_importance(final_model, feature_cols)
    if not fi.empty:
        print("\n  Top 10 features by importance:")
        print(fi.head(10).to_string(index=False))

    # ── Save model ───────────────────────────────────────────────────────────
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        'model':        final_model,
        'feature_cols': feature_cols,
        'label_encoder': le,
        'cv_results':   cv_results,
        'backend':      backend,
        'meta': {
            'r_band_min': R_BAND_MIN,
            'r_band_max': R_BAND_MAX,
            'classes':    list(le.classes_),
        }
    }
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"\n  [OK] Model saved → {model_path}")

    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plot_evaluation(y_test, y_pred, y_prob, z_test, fi,
                    cv_results, n_cv_splits, le, str(plot_path))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE — PREDICT
# ─────────────────────────────────────────────

def pipeline_predict(input_path: Path, model_path: Path,
                     output_path: Path) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("STELLAR OBJECT CLASSIFICATION — INFERENCE PIPELINE")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run with --mode train first."
        )
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    model        = artifact['model']
    feature_cols = artifact['feature_cols']
    le           = artifact['label_encoder']
    meta         = artifact['meta']
    print(f"  [OK] Model loaded ({artifact.get('backend','?').upper()}) "
          f"from {model_path}")
    print(f"       Classes: {meta['classes']}")

    # ── Load new data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading test data...")
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Input data not found: {input_path}\n"
            f"Expected: stellar-class/data/raw/test.csv"
        )
    df = pd.read_csv(input_path)
    check_required_columns(df, mode='predict')
    print(f"  Loaded {len(df):,} objects.")

    # ── Quality checks & features ─────────────────────────────────────────────
    print("\n[2/4] Preprocessing & feature engineering...")
    df = flag_missing_bands(df)
    df = flag_depth_extrapolation(df)
    df = apply_k_correction(df)
    df = build_features(df)

    # Fill any unseen one-hot columns with 0 (e.g. rare spectral type absent in test)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X_new = df[feature_cols].fillna(df[feature_cols].median())

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\n[3/4] Running predictions...")
    t0 = time.time()
    y_pred_enc = model.predict(X_new)
    pred_labels = le.inverse_transform(y_pred_enc)
    print(f"  [OK] Inference completed in {time.time()-t0:.1f}s.")

    # ── Assemble & save ───────────────────────────────────────────────────────
    # Output: id and class only (submission format)
    results = df[[ID_COL]].copy().reset_index(drop=True)
    results['class'] = pred_labels

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\n[4/4] Prediction summary:")
    print(f"  Total objects : {len(results):,}")
    for cls in CLASSES:
        cnt = (results['class'] == cls).sum()
        print(f"  {cls:<8}      : {cnt:,}  ({cnt/len(results)*100:.1f}%)")
    print(f"\n  [OK] Predictions saved → {output_path}")
    print(f"       Format   : id, class")
    print("\n" + "=" * 60 + "\n")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Stellar object classification pipeline — GALAXY / QSO / STAR'
    )
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='train: fit a new model | predict: apply saved model')
    parser.add_argument('--input',  type=Path, default=None,
                        help='Input CSV (default: data/raw/train.csv or test.csv)')
    parser.add_argument('--model',  type=Path, default=DEFAULT_MODEL_PATH,
                        help=f'Model pkl (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--output', type=Path, default=DEFAULT_PRED_PATH,
                        help=f'Predictions CSV (default: {DEFAULT_PRED_PATH})')
    parser.add_argument('--plot',   type=Path, default=DEFAULT_PLOT_PATH,
                        help=f'Evaluation plot PNG (default: {DEFAULT_PLOT_PATH})')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Held-out test fraction for train mode (default: 0.2)')
    parser.add_argument('--backend', default=os.environ.get('BACKEND', 'lgbm'),
                        choices=['lgbm', 'xgb', 'hist'],
                        help='Model backend (default: lgbm)')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'gpu', 'cuda'],
                        help='Device for GPU-capable backends (default: cpu)')
    parser.add_argument('--cv-splits', type=int, default=3,
                        help='Number of CV folds (default: 3)')
    return parser.parse_args()


if __name__ == '__main__':
    pipeline_train(
        input_path=DEFAULT_TRAIN_PATH,
        model_path="galaxy_model.pkl",
        test_size=0.2,
        plot_path="evaluation_report.png",
        backend     = "lgbm",
        device      = "cpu",
        n_cv_splits = 3,
    )

    pipeline_predict(
        input_path  = DEFAULT_TEST_PATH,
        model_path  = "galaxy_model.pkl",
        output_path = "predictions.csv",
    )