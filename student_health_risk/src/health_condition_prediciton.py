"""
health_condition_pipeline.py
────────────────────────────
End-to-end pipeline: load → feature engineer → train → validate → predict.

Usage
-----
    python health_condition_pipeline.py

Edit the CONFIG block below to point at your files or tune any setting.
Output: submission.csv  with columns  id | health_condition
"""

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Edit these paths and settings; everything else runs automatically.

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH  = os.path.join(BASE_DIR, "data", "raw", "train.csv")
TEST_PATH   = os.path.join(BASE_DIR, "data", "raw", "test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "submissions", "submission.csv")

N_SPLITS     = 5       # number of CV folds
SEED         = 42
LGBM_WEIGHT  = 0.5     # blend weight for LightGBM (XGBoost gets 1 - this)

# ── COLUMN DEFINITIONS ────────────────────────────────────────────────────────

TARGET_COL = "health_condition"
ID_COL     = "id"

NUMERIC_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

CATEGORICAL_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Activity composite
    df["activity_score"]   = df["step_count"] + (df["exercise_duration"] * 100)
    df["calorie_per_step"] = df["calorie_expenditure"] / (df["step_count"] + 1)
    df["hydration_ratio"]  = df["water_intake"] / (df["exercise_duration"] + 1)
    df["sleep_bmi_ratio"]  = df["sleep_duration"] / (df["bmi"] + 1)

    # BMI clinical bands
    df["bmi_band"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25.0, 30.0, 100.0],
        labels=["underweight", "normal", "overweight", "obese"],
    ).astype(str)

    # Heart rate zones
    df["hr_zone"] = pd.cut(
        df["heart_rate"],
        bins=[0, 60, 80, 100, 300],
        labels=["low", "normal", "elevated", "high"],
    ).astype(str)

    return df


def encode_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all remaining string/object columns to integers."""
    df = df.copy()
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if str_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[str_cols] = enc.fit_transform(df[str_cols].astype(str))
    return df


def build_features(
    df: pd.DataFrame,
    is_train: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None, LabelEncoder | None]:
    """
    Full feature pipeline.

    Returns
    -------
    X   : feature dataframe (no id, no target)
    y   : encoded target series (None for test)
    le  : fitted LabelEncoder (None for test)
    """
    ids = df[ID_COL].copy()
    df  = df.drop(columns=[ID_COL], errors="ignore")

    df = fill_missing_values(df)
    df = add_features(df)

    le, y = None, None
    if is_train and TARGET_COL in df.columns:
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(df[TARGET_COL]), name="target")
        df = df.drop(columns=[TARGET_COL])

    elif not is_train and TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    df = encode_strings(df)

    return df, y, le, ids


# ── MODEL FACTORIES ───────────────────────────────────────────────────────────

def lgbm_factory():
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {
        "callbacks": [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    }
    return model, fit_kwargs


def xgb_factory():
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        enable_categorical=True,
        tree_method="hist",
        verbosity=0,
    )
    fit_kwargs = {}
    return model, fit_kwargs


# ── CROSS-VALIDATION ──────────────────────────────────────────────────────────

def cross_validate(
    model_fn,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified K-Fold CV.
    Returns OOF predictions and averaged test predictions.
    """
    n_classes  = y.nunique()
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model, fit_kwargs = model_fn()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], **fit_kwargs)

        oof_preds[val_idx]  = model.predict_proba(X_val)
        test_preds         += model.predict_proba(X_test) / N_SPLITS

        score = accuracy_score(y_val, oof_preds[val_idx].argmax(axis=1))
        print(f"    Fold {fold + 1}/{N_SPLITS}  accuracy: {score:.5f}")

    oof_score = accuracy_score(y, oof_preds.argmax(axis=1))
    print(f"    OOF accuracy : {oof_score:.5f}\n")
    return oof_preds, test_preds


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  Health Condition Prediction Pipeline")
    print("=" * 55)
    print(f"\n[1/5] Loading data ...")

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)

    print(f"  Train : {train_raw.shape}")
    print(f"  Test  : {test_raw.shape}")
    print(f"  Target classes : {sorted(train_raw[TARGET_COL].unique())}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print(f"\n[2/5] Building features ...")

    X,      y,  le, _        = build_features(train_raw, is_train=True)
    X_test, _,  _,  test_ids = build_features(test_raw,  is_train=False)

    print(f"  Feature count : {X.shape[1]}")
    print(f"  Label mapping : {dict(enumerate(le.classes_))}")

    # ── 3. Train LightGBM ─────────────────────────────────────────────────────
    print(f"\n[3/5] Training LightGBM ({N_SPLITS}-fold CV) ...")
    lgbm_oof, lgbm_test = cross_validate(lgbm_factory, X, y, X_test)

    # ── 4. Train XGBoost ─────────────────────────────────────────────────────
    print(f"[4/5] Training XGBoost ({N_SPLITS}-fold CV) ...")
    xgb_oof, xgb_test = cross_validate(xgb_factory, X, y, X_test)

    # ── 5. Blend + evaluate ───────────────────────────────────────────────────
    print(f"[5/5] Blending predictions ...")

    xgb_weight = round(1 - LGBM_WEIGHT, 2)
    blend_oof  = lgbm_oof  * LGBM_WEIGHT + xgb_oof  * xgb_weight
    blend_test = lgbm_test * LGBM_WEIGHT + xgb_test * xgb_weight

    lgbm_score  = accuracy_score(y, lgbm_oof.argmax(axis=1))
    xgb_score   = accuracy_score(y, xgb_oof.argmax(axis=1))
    blend_score = accuracy_score(y, blend_oof.argmax(axis=1))

    print(f"\n  LGBM  OOF accuracy : {lgbm_score:.5f}")
    print(f"  XGB   OOF accuracy : {xgb_score:.5f}")
    print(f"  Blend OOF accuracy : {blend_score:.5f}  "
          f"(LGBM×{LGBM_WEIGHT} + XGB×{xgb_weight})")

    # Classification report on OOF predictions
    print(f"\n{'─' * 55}")
    print("  Validation report (out-of-fold predictions)")
    print(f"{'─' * 55}")
    print(classification_report(
        y,
        blend_oof.argmax(axis=1),
        target_names=le.classes_,
        digits=4,
    ))

    # ── 6. Save submission ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    pred_labels = le.inverse_transform(blend_test.argmax(axis=1))

    submission = pd.DataFrame({
        ID_COL:     test_ids,
        TARGET_COL: pred_labels,
    })

    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"  Submission saved  : {OUTPUT_PATH}")
    print(f"  Rows              : {len(submission)}")
    print(f"\n  Predicted class distribution:")
    dist = submission[TARGET_COL].value_counts()
    for cls, count in dist.items():
        pct = count / len(submission) * 100
        print(f"    {cls:<20} {count:>6}  ({pct:.1f}%)")

    print(f"\n{'=' * 55}")
    print("  Done.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()