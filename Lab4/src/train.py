"""
train.py
--------
Trains a LightGBM binary classifier to predict next-day
S&P 500 price direction (Up=1 / Down=0).

Uses TimeSeriesSplit for chronologically correct cross-validation
(no data leakage — future data never trains on past folds).

Saves:
  - models/lgbm_model.pkl   → trained LightGBM model
  - models/feature_cols.pkl → list of feature column names
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)
from features import FEATURE_COLS, TARGET_COL


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")
FEAT_PATH  = os.path.join(MODEL_DIR, "feature_cols.pkl")


def train(df: pd.DataFrame):
    """
    Train LightGBM classifier with time-series cross-validation.
    Saves the model trained on the full dataset after CV evaluation.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f"\n📊 Dataset — Samples: {len(X)} | Features: {len(FEATURE_COLS)}")
    print(f"   Class balance → Up: {y.sum()} | Down: {(y==0).sum()}\n")

    # ── Time-Series Cross Validation (5 folds) ──────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accs, fold_aucs = [], []

    params = {
        "objective":      "binary",
        "metric":         "binary_logloss",
        "n_estimators":   500,
        "learning_rate":  0.03,
        "num_leaves":     31,
        "max_depth":      -1,
        "min_child_samples": 20,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":      0.1,
        "reg_lambda":     0.1,
        "random_state":   42,
        "verbose":        -1,
    }

    print("⚙️  Running TimeSeriesSplit Cross-Validation (5 folds)...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )

        preds     = model.predict(X_val)
        probs     = model.predict_proba(X_val)[:, 1]
        acc       = accuracy_score(y_val, preds)
        auc       = roc_auc_score(y_val, probs)
        fold_accs.append(acc)
        fold_aucs.append(auc)
        print(f"   Fold {fold} → Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    print(f"\n📈 CV Results → Avg Accuracy: {np.mean(fold_accs):.4f} "
          f"| Avg ROC-AUC: {np.mean(fold_aucs):.4f}")

    # ── Final Model — train on full data ────────────────────────────────────
    print("\n⚙️  Training final model on full dataset...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

    # ── Hold-out evaluation (last 20% as test) ──────────────────────────────
    split = int(len(X) * 0.8)
    X_test, y_test = X.iloc[split:], y.iloc[split:]
    test_preds = final_model.predict(X_test)
    test_probs = final_model.predict_proba(X_test)[:, 1]

    print("\n📋 Hold-Out Test Set Evaluation (last 20% chronologically):")
    print(classification_report(y_test, test_preds,
                                target_names=["Down (0)", "Up (1)"]))
    print(f"   ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")

    # ── Save model & feature list ────────────────────────────────────────────
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(FEATURE_COLS, FEAT_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")
    print(f"💾 Feature list saved → {FEAT_PATH}")

    return final_model