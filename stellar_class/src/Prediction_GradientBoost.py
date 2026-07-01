"""
Galaxy Population Prediction Pipeline
======================================
Predicts galaxy population (Red_Sequence / Blue_Cloud) from photometric
survey data using Gradient Boosted Decision Trees.

NOTE ON XGBOOST
---------------
This script uses sklearn.ensemble.GradientBoostingClassifier, which
implements the same GBDT algorithm as XGBoost. To switch to XGBoost:
    pip install xgboost
    # Then replace the import and MODEL block below:
    from xgboost import XGBClassifier
    MODEL_PARAMS = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric='logloss', random_state=42
    )
    model = XGBClassifier(**MODEL_PARAMS)

USAGE
-----
    # Train on labelled data and save model:
    python galaxy_prediction_pipeline.py --mode train --input analyse.csv

    # Predict on new unlabelled data:
    python galaxy_prediction_pipeline.py --mode predict --input new_survey.csv --model galaxy_model.pkl

    # Train + immediately evaluate on a held-out test split:
    python galaxy_prediction_pipeline.py --mode train --input analyse.csv --test-size 0.2

INPUT FORMAT
------------
Required columns: id, alpha, delta, u, g, r, i, z, redshift, spectral_type
Label column (train mode only): galaxy_population  ('Red_Sequence' or 'Blue_Cloud')
"""

import argparse
import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize
from pathlib import Path

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LABEL_COL        = 'galaxy_population'
POSITIVE_CLASS   = 'Red_Sequence'
NEGATIVE_CLASS   = 'Blue_Cloud'
SPECTRAL_COL     = 'spectral_type'
BAND_COLS        = ['u', 'g', 'r', 'i', 'z']
COORD_COLS       = ['alpha', 'delta']
REDSHIFT_COL     = 'redshift'
ID_COL           = 'id'

# Anomalous label combinations identified during EDA
ANOMALOUS_COMBOS = [
    {'spectral_type': 'O/B',  'galaxy_population': 'Red_Sequence'},
    {'spectral_type': 'A/F',  'galaxy_population': 'Red_Sequence'},
]

# Redshift bins for stratified evaluation
Z_BINS   = [0, 0.2, 0.5, 1.0, 2.0, 99]
Z_LABELS = ['<0.2', '0.2–0.5', '0.5–1', '1–2', '>2']

# Training magnitude range from base dataset (flag extrapolations)
R_BAND_MIN, R_BAND_MAX = 14.85, 23.13

MODEL_PARAMS = dict(
    n_estimators  = 300,
    max_depth     = 4,
    learning_rate = 0.05,
    subsample     = 0.8,
    min_samples_leaf = 5,
    random_state  = 42,
)


# ─────────────────────────────────────────────
# PHASE 1 — DATA QUALITY & PREPROCESSING
# ─────────────────────────────────────────────

def check_required_columns(df: pd.DataFrame, mode: str) -> None:
    """Raise informative errors if required columns are missing."""
    required = [ID_COL, SPECTRAL_COL, REDSHIFT_COL] + BAND_COLS + COORD_COLS
    if mode == 'train':
        required.append(LABEL_COL)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def flag_missing_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows with missing values in photometric bands."""
    missing_mask = df[BAND_COLS + [REDSHIFT_COL]].isnull().any(axis=1)
    n_missing = missing_mask.sum()
    if n_missing > 0:
        print(f"  [WARNING] {n_missing} rows have missing band/redshift values "
              f"— these will be kept but flagged in output.")
    df = df.copy()
    df['flag_missing_bands'] = missing_mask
    return df


def apply_k_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a first-order K-correction proxy.

    True K-corrections require SED templates (e.g. LePhare, EAZY).
    This proxy adjusts each band by a linear approximation based on
    redshift, matching the slope of the known K-correction curves
    for typical galaxy SEDs in SDSS-like filters.

    Replace this function with full template-fitting corrections in
    production — the coefficient dict below is illustrative.
    """
    # Approximate correction slopes  Δm ≈ k_coeff × redshift  (mag)
    k_coeffs = {'u': 1.0, 'g': 0.6, 'r': 0.3, 'i': 0.15, 'z': 0.05}
    df = df.copy()
    for band, coeff in k_coeffs.items():
        df[f'{band}_kcorr'] = df[band] - coeff * df[REDSHIFT_COL]
    print("  [OK] K-correction proxy applied to all 5 bands.")
    return df


def remove_anomalous_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove physically implausible spectral-type + population combos.
    EDA identified O/B+Red_Sequence (1 obj) and A/F+Red_Sequence (25 obj).
    These are likely misclassifications or stellar contaminants.
    """
    if LABEL_COL not in df.columns:
        return df
    mask = pd.Series(False, index=df.index)
    for combo in ANOMALOUS_COMBOS:
        m = pd.Series(True, index=df.index)
        for col, val in combo.items():
            m &= (df[col] == val)
        mask |= m
    n_removed = mask.sum()
    if n_removed > 0:
        print(f"  [OK] Removed {n_removed} anomalous label rows "
              f"({', '.join(str(c) for c in ANOMALOUS_COMBOS)}).")
    return df[~mask].copy()


def flag_high_z_outliers(df: pd.DataFrame, threshold: float = 1.92) -> pd.DataFrame:
    """
    Flag high-redshift outliers (z > 1.92 based on IQR analysis of training data).
    These may include AGN/QSOs. Flagged rows are predicted but marked for review.
    """
    df = df.copy()
    df['flag_high_z'] = df[REDSHIFT_COL] > threshold
    n_flagged = df['flag_high_z'].sum()
    if n_flagged > 0:
        print(f"  [WARNING] {n_flagged} objects flagged as high-z outliers (z > {threshold}). "
              f"Predictions for these should be reviewed — possible AGN/QSO contamination.")
    return df


def flag_depth_extrapolation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag objects outside the training magnitude range in r-band.
    These are extrapolations the model has not seen during training.
    """
    df = df.copy()
    out_of_range = (df['r'] < R_BAND_MIN) | (df['r'] > R_BAND_MAX)
    df['flag_depth_extrapolation'] = out_of_range
    n_flagged = out_of_range.sum()
    if n_flagged > 0:
        print(f"  [WARNING] {n_flagged} objects outside training r-band range "
              f"[{R_BAND_MIN:.2f}, {R_BAND_MAX:.2f}] — flagged as extrapolations.")
    return df


# ─────────────────────────────────────────────
# PHASE 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all predictive features from raw columns.

    Features used:
    - Colour indices (primary discriminators): u−g, g−r, r−i, i−z
      (and their K-corrected equivalents)
    - Photometric redshift
    - Sky coordinates (alpha, delta)
    - Spectral type (one-hot encoded — not ordinal)

    Raw magnitudes are NOT used directly because:
    - Adjacent bands are collinear (r–i r=0.954, i–z r=0.967)
    - Raw magnitudes conflate distance with intrinsic brightness
    - Colour indices remove the common-mode distance component
    """
    df = df.copy()

    # Standard colour indices from raw bands
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']

    # K-corrected colour indices (if k-correction was applied)
    kcorr_cols = [f'{b}_kcorr' for b in BAND_COLS]
    if all(c in df.columns for c in kcorr_cols):
        df['u_g_k'] = df['u_kcorr'] - df['g_kcorr']
        df['g_r_k'] = df['g_kcorr'] - df['r_kcorr']
        df['r_i_k'] = df['r_kcorr'] - df['i_kcorr']
        df['i_z_k'] = df['i_kcorr'] - df['z_kcorr']

    # One-hot encode spectral type (4 categories: M, A/F, G/K, O/B)
    spec_dummies = pd.get_dummies(df[SPECTRAL_COL], prefix='spec', dtype=float)
    df = pd.concat([df, spec_dummies], axis=1)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the ordered list of feature column names used for modelling."""
    base_features = ['u_g', 'g_r', 'r_i', 'i_z', REDSHIFT_COL] + COORD_COLS
    kcorr_features = ['u_g_k', 'g_r_k', 'r_i_k', 'i_z_k'] \
        if 'u_g_k' in df.columns else []
    spec_features = [c for c in df.columns if c.startswith('spec_')]
    return base_features + kcorr_features + spec_features


def encode_labels(series: pd.Series) -> tuple:
    """Encode galaxy_population to binary int. Returns (y_encoded, encoder_dict)."""
    mapping = {POSITIVE_CLASS: 1, NEGATIVE_CLASS: 0}
    encoded = series.map(mapping)
    if encoded.isnull().any():
        unknown = series[encoded.isnull()].unique()
        raise ValueError(f"Unknown label values: {unknown}. "
                         f"Expected '{POSITIVE_CLASS}' or '{NEGATIVE_CLASS}'.")
    return encoded.astype(int), {v: k for k, v in mapping.items()}


# ─────────────────────────────────────────────
# PHASE 3 — MODEL TRAINING
# ─────────────────────────────────────────────

def build_model(calibrate: bool = True):
    """
    Build the gradient boosted classifier.

    Calibration (Platt scaling) is applied by default to produce
    well-calibrated probability estimates. A hard label of 0.50 threshold
    is used for classification, but downstream users should threshold
    at their desired confidence level.

    To use XGBoost instead, replace GradientBoostingClassifier with:
        from xgboost import XGBClassifier
        base = XGBClassifier(**MODEL_PARAMS, eval_metric='logloss')
    """
    base = GradientBoostingClassifier(**MODEL_PARAMS)
    if calibrate:
        # isotonic regression calibration via 5-fold internal CV
        return CalibratedClassifierCV(base, method='isotonic', cv=3)
    return base


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series,
                          n_splits: int = 5) -> dict:
    """Run stratified k-fold CV and return mean ± std of key metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
        return_train_score=True
    )
    results = {}
    for metric in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
        test_scores = scores[f'test_{metric}']
        results[metric] = {
            'mean': test_scores.mean(),
            'std':  test_scores.std(),
            'values': test_scores.tolist()
        }
    return results


# ─────────────────────────────────────────────
# PHASE 4 — EVALUATION
# ─────────────────────────────────────────────

def evaluate_by_redshift_bin(y_true: pd.Series, y_pred: np.ndarray,
                              y_prob: np.ndarray,
                              z_values: pd.Series) -> pd.DataFrame:
    """
    Compute classification metrics separately per redshift bin.

    This is critical: population mix flips from RS-dominated at z<0.5
    to BC-dominated at z>1. A single accuracy number masks this shift.
    """
    z_bins = pd.cut(z_values, bins=Z_BINS, labels=Z_LABELS)
    rows = []
    for label in Z_LABELS:
        mask = (z_bins == label)
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_prob[mask]
        n = mask.sum()
        acc = (yt == yp).mean()
        try:
            auc = roc_auc_score(yt, ypr) if yt.nunique() > 1 else float('nan')
        except Exception:
            auc = float('nan')
        rs_frac = yt.mean()
        rows.append({
            'Redshift bin':   label,
            'N objects':      n,
            'RS fraction':    f'{rs_frac:.2f}',
            'Accuracy':       f'{acc:.4f}',
            'AUC-ROC':        f'{auc:.4f}' if not np.isnan(auc) else 'N/A',
        })
    return pd.DataFrame(rows)


def compute_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from the trained model."""
    # CalibratedClassifierCV wraps the base estimator
    base = model.estimators_[0].estimator \
        if hasattr(model, 'estimators_') else model
    if hasattr(base, 'feature_importances_'):
        importances = base.feature_importances_
    else:
        # Fallback for CalibratedClassifierCV
        try:
            importances = np.mean([
                e.estimator.feature_importances_
                for e in model.calibrated_classifiers_
            ], axis=0)
        except Exception:
            return pd.DataFrame()
    df_imp = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    return df_imp


# ─────────────────────────────────────────────
# PHASE 5 — VISUALISATION
# ─────────────────────────────────────────────

def plot_evaluation(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray,
                    z_test: pd.Series, feature_importance: pd.DataFrame,
                    cv_results: dict, output_path: str = 'evaluation_report.png') -> None:
    """
    Generate a 6-panel evaluation figure:
      1. Confusion matrix
      2. ROC curve
      3. Calibration curve (reliability diagram)
      4. Probability distribution by true class
      5. Redshift-bin accuracy bar chart
      6. Feature importances
    """
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    BLUE   = '#185FA5'
    CORAL  = '#D85A30'
    PURPLE = '#534AB7'
    TEAL   = '#1D9E75'
    GRAY   = '#888780'

    label_map = {1: POSITIVE_CLASS, 0: NEGATIVE_CLASS}

    # ── Panel 1: Confusion matrix ──
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[NEGATIVE_CLASS, POSITIVE_CLASS]
    )
    disp.plot(ax=ax1, cmap='Blues', colorbar=False)
    ax1.set_title('Confusion matrix', fontsize=12, fontweight='medium', pad=10)
    ax1.tick_params(labelsize=9)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

    # ── Panel 2: ROC curve ──
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    ax2.plot(fpr, tpr, color=BLUE, lw=2, label=f'AUC = {auc_score:.4f}')
    ax2.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle='--', label='Random')
    ax2.set_xlabel('False positive rate', fontsize=10)
    ax2.set_ylabel('True positive rate', fontsize=10)
    ax2.set_title('ROC curve', fontsize=12, fontweight='medium', pad=10)
    ax2.legend(fontsize=9)
    ax2.set_facecolor('#F8F8F8')
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Calibration curve ──
    ax3 = fig.add_subplot(gs[0, 2])
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax3.plot(prob_pred, prob_true, marker='o', color=TEAL, lw=2,
             label='Model', markersize=5)
    ax3.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle='--', label='Perfect')
    ax3.set_xlabel('Mean predicted probability', fontsize=10)
    ax3.set_ylabel('Fraction of positives', fontsize=10)
    ax3.set_title('Calibration curve', fontsize=12, fontweight='medium', pad=10)
    ax3.legend(fontsize=9)
    ax3.set_facecolor('#F8F8F8')
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Predicted probability distribution ──
    ax4 = fig.add_subplot(gs[1, 0])
    for cls_int, color, cls_name in [(1, CORAL, POSITIVE_CLASS),
                                      (0, BLUE,  NEGATIVE_CLASS)]:
        mask = y_test == cls_int
        ax4.hist(y_prob[mask], bins=20, alpha=0.6, color=color,
                 label=cls_name.replace('_', ' '), density=True)
    ax4.axvline(0.5, color=GRAY, lw=1.5, linestyle='--', label='Threshold 0.5')
    ax4.set_xlabel('Predicted probability (Red Sequence)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.set_title('Score distribution by true class', fontsize=12, fontweight='medium', pad=10)
    ax4.legend(fontsize=8)
    ax4.set_facecolor('#F8F8F8')
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Accuracy by redshift bin ──
    ax5 = fig.add_subplot(gs[1, 1])
    z_bins_col = pd.cut(z_test, bins=Z_BINS, labels=Z_LABELS)
    bin_acc = []
    bin_labels_present = []
    bin_counts = []
    for label in Z_LABELS:
        mask = (z_bins_col == label)
        if mask.sum() == 0:
            continue
        acc = (y_test[mask] == y_pred[mask]).mean()
        bin_acc.append(acc)
        bin_labels_present.append(label)
        bin_counts.append(mask.sum())

    bars = ax5.bar(bin_labels_present, bin_acc, color=PURPLE, alpha=0.75,
                   edgecolor='white', width=0.6)
    for bar, n in zip(bars, bin_counts):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'n={n}', ha='center', va='bottom', fontsize=8, color=GRAY)
    ax5.set_ylim(0, 1.1)
    ax5.set_xlabel('Redshift bin', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Accuracy by redshift bin', fontsize=12, fontweight='medium', pad=10)
    ax5.axhline(1.0, color=GRAY, lw=0.8, linestyle='--')
    ax5.set_facecolor('#F8F8F8')
    ax5.grid(True, alpha=0.3, axis='y')

    # ── Panel 6: Feature importances ──
    ax6 = fig.add_subplot(gs[1, 2])
    if not feature_importance.empty:
        top_n = feature_importance.head(12)
        colors = [CORAL if 'u_g' in f or 'g_r' in f or 'r_i' in f or 'i_z' in f
                  else BLUE if 'redshift' in f
                  else TEAL if 'spec' in f
                  else GRAY
                  for f in top_n['Feature']]
        ax6.barh(top_n['Feature'][::-1], top_n['Importance'][::-1],
                 color=colors[::-1], alpha=0.8, edgecolor='white')
        ax6.set_xlabel('Importance', fontsize=10)
        ax6.set_title('Feature importances (top 12)', fontsize=12,
                      fontweight='medium', pad=10)
        ax6.set_facecolor('#F8F8F8')
        ax6.grid(True, alpha=0.3, axis='x')
        ax6.tick_params(labelsize=8)

    fig.suptitle('Galaxy Population Classifier — Evaluation Report',
                 fontsize=15, fontweight='medium', y=1.01)

    cv_acc  = cv_results['accuracy']['mean']
    cv_auc  = cv_results['roc_auc']['mean']
    cv_f1   = cv_results['f1']['mean']
    subtitle = (f'5-fold CV:  Accuracy {cv_acc:.4f}  |  '
                f'AUC {cv_auc:.4f}  |  F1 {cv_f1:.4f}')
    fig.text(0.5, 0.995, subtitle, ha='center', va='bottom',
             fontsize=10, color=GRAY)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Evaluation plots saved → {output_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def pipeline_train(input_path: str, model_path: str = 'galaxy_model.pkl',
                   test_size: float = 0.2,
                   plot_path: str = 'evaluation_report.png') -> None:
    """
    Full training pipeline:
      1. Load & validate data
      2. Quality checks & anomaly removal
      3. K-correction proxy
      4. Feature engineering
      5. Cross-validation
      6. Final model fit + calibration
      7. Held-out test evaluation
      8. Save model + generate report
    """
    print("\n" + "="*60)
    print("GALAXY POPULATION PREDICTION — TRAINING PIPELINE")
    print("="*60)

    # ── Load ──
    print("\n[1/7] Loading data...")
    df = pd.read_csv(input_path)
    check_required_columns(df, mode='train')
    print(f"  Loaded {len(df)} rows × {df.shape[1]} columns.")

    # ── Quality checks ──
    print("\n[2/7] Data quality checks...")
    df = flag_missing_bands(df)
    df = remove_anomalous_labels(df)
    df = flag_high_z_outliers(df)
    df = flag_depth_extrapolation(df)

    # ── K-correction ──
    print("\n[3/7] Applying K-correction proxy...")
    df = apply_k_correction(df)

    # ── Feature engineering ──
    print("\n[4/7] Engineering features...")
    df = build_features(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(df[feature_cols].median())
    y, label_decoder = encode_labels(df[LABEL_COL])
    print(f"  Feature set ({len(feature_cols)}): {feature_cols}")
    print(f"  Class distribution — RS: {y.sum()}  BC: {(y==0).sum()}")

    # ── Cross-validation ──
    print("\n[5/7] Cross-validating (5-fold stratified)...")
    model_cv = build_model(calibrate=True)
    cv_results = cross_validate_model(model_cv, X, y, n_splits=5)
    print("\n  ┌─────────────────────────────────────────────┐")
    print("  │          Cross-Validation Results           │")
    print("  ├──────────────┬──────────┬───────────────────┤")
    print(f"  │ {'Metric':<12} │ {'Mean':>8} │ {'Std':>17} │")
    print("  ├──────────────┼──────────┼───────────────────┤")
    for metric, vals in cv_results.items():
        print(f"  │ {metric:<12} │ {vals['mean']:>8.4f} │ {vals['std']:>17.4f} │")
    print("  └──────────────┴──────────┴───────────────────┘")

    # ── Train-test split & final fit ──
    print(f"\n[6/7] Training final model (test split {test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    z_test = df.loc[y_test.index, REDSHIFT_COL]

    final_model = build_model(calibrate=True)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    # ── Evaluation ──
    print("\n[7/7] Evaluating on held-out test set...")
    print("\n  Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[NEGATIVE_CLASS, POSITIVE_CLASS]
    ))

    print("  Accuracy by redshift bin:")
    z_report = evaluate_by_redshift_bin(y_test, y_pred, y_prob, z_test)
    print(z_report.to_string(index=False))

    fi = compute_feature_importance(final_model, feature_cols)
    if not fi.empty:
        print("\n  Top 10 features by importance:")
        print(fi.head(10).to_string(index=False))

    # ── Save ──
    artifact = {
        'model':          final_model,
        'feature_cols':   feature_cols,
        'label_decoder':  label_decoder,
        'cv_results':     cv_results,
        'meta': {
            'r_band_min': R_BAND_MIN,
            'r_band_max': R_BAND_MAX,
            'z_outlier_threshold': 1.92,
            'positive_class': POSITIVE_CLASS,
        }
    }
    with open(model_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"\n  [OK] Model saved → {model_path}")

    plot_evaluation(y_test, y_pred, y_prob, z_test, fi, cv_results, plot_path)
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")


def pipeline_predict(input_path: str, model_path: str = 'galaxy_model.pkl',
                     output_path: str = 'predictions.csv') -> pd.DataFrame:
    """
    Prediction pipeline for new (unlabelled) survey data.
    Applies the same preprocessing as training, then outputs:
      - predicted_label       (Red_Sequence / Blue_Cloud)
      - predicted_probability (probability of Red_Sequence)
      - confidence_tier       (High / Medium / Low)
      - flag_high_z, flag_depth_extrapolation, flag_missing_bands
    """
    print("\n" + "="*60)
    print("GALAXY POPULATION PREDICTION — INFERENCE PIPELINE")
    print("="*60)

    # ── Load model ──
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    model         = artifact['model']
    feature_cols  = artifact['feature_cols']
    label_decoder = artifact['label_decoder']
    meta          = artifact['meta']
    print(f"  [OK] Model loaded from {model_path}")

    # ── Load new data ──
    print("\n[1/5] Loading new survey data...")
    df = pd.read_csv(input_path)
    check_required_columns(df, mode='predict')
    print(f"  Loaded {len(df)} objects.")

    # ── Quality flags ──
    print("\n[2/5] Quality checks...")
    df = flag_missing_bands(df)
    df = flag_high_z_outliers(df, threshold=meta['z_outlier_threshold'])
    df = flag_depth_extrapolation(df)

    # ── K-correction & features ──
    print("\n[3/5] K-correction & feature engineering...")
    df = apply_k_correction(df)
    df = build_features(df)

    # Ensure all expected columns are present (fill unseen spec types with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    X_new = df[feature_cols].fillna(df[feature_cols].median())

    # ── Predict ──
    print("\n[4/5] Running predictions...")
    y_prob = model.predict_proba(X_new)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    pred_labels = pd.Series(y_pred).map(label_decoder)

    # Confidence tier
    confidence = pd.cut(
        np.maximum(y_prob, 1 - y_prob),
        bins=[0.5, 0.7, 0.9, 1.01],
        labels=['Low', 'Medium', 'High']
    )

    # ── Assemble output ──
    results = df[[ID_COL]].copy()
    results['predicted_label']       = pred_labels.values
    results['predicted_probability'] = y_prob.round(4)
    results['confidence_tier']       = confidence.values
    results['flag_high_z']           = df['flag_high_z'].values
    results['flag_depth_extrap']     = df['flag_depth_extrapolation'].values
    results['flag_missing_bands']    = df['flag_missing_bands'].values

    results.to_csv(output_path, index=False)

    print(f"\n[5/5] Prediction summary:")
    print(f"  Total objects:    {len(results)}")
    print(f"  Red_Sequence:     {(results['predicted_label']==POSITIVE_CLASS).sum()}")
    print(f"  Blue_Cloud:       {(results['predicted_label']==NEGATIVE_CLASS).sum()}")
    print(f"  High confidence:  {(results['confidence_tier']=='High').sum()}")
    print(f"  Medium confidence:{(results['confidence_tier']=='Medium').sum()}")
    print(f"  Low confidence:   {(results['confidence_tier']=='Low').sum()}")
    print(f"  Flagged (high-z): {results['flag_high_z'].sum()}")
    print(f"  Flagged (depth):  {results['flag_depth_extrap'].sum()}")
    print(f"\n  [OK] Predictions saved → {output_path}")
    print("\n" + "="*60 + "\n")

    return results


# ─────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Galaxy population prediction pipeline (GBT / XGBoost equivalent)'
    )
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='train: fit a new model | predict: apply existing model')
    parser.add_argument('--input', required=True,
                        help='Path to input CSV (training or new survey data)')
    parser.add_argument('--model', default='galaxy_model.pkl',
                        help='Path to save (train) or load (predict) the model')
    parser.add_argument('--output', default='predictions.csv',
                        help='Path for prediction output CSV (predict mode only)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Held-out test fraction for train mode (default: 0.2)')
    parser.add_argument('--plot', default='evaluation_report.png',
                        help='Path for evaluation plot PNG (train mode only)')
    return parser.parse_args()


if __name__ == '__main__':
    # This always points to stellar-class/src/predict.py
    SRC_DIR  = Path(__file__).resolve().parent        # stellar-class/src/
    ROOT_DIR = SRC_DIR.parent                          # stellar-class/

    # Now all paths are stable regardless of where you run from
    RAW_DIR         = ROOT_DIR / "data" / "raw"
    PROCESSED_DIR   = ROOT_DIR / "data" / "processed"
    PREDICTIONS_DIR = ROOT_DIR / "data" / "predictions"
    SUBMISSIONS_DIR = ROOT_DIR / "data" / "submissions"

    # Usage
    train_path = RAW_DIR / "train.csv"
    test_path  = RAW_DIR / "test.csv"
    pipeline_train(
        input_path=train_path,
        model_path="galaxy_model.pkl",
        test_size=0.2,
        plot_path="evaluation_report.png"
    )
    
    pipeline_predict(
            input_path=test_path,
            model_path="galaxy_model.pkl",
            output_path="predictions.csv"
        )