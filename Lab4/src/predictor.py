"""
predict.py
----------
Loads the saved LightGBM model and runs inference on the most
recent trading day's features to predict tomorrow's S&P 500 direction.

Output:
  - Predicted direction: UP ↑ or DOWN ↓
  - Confidence probability
  - Latest feature snapshot used for prediction
"""

import joblib
import pandas as pd
from features import fetch_data, engineer_features, FEATURE_COLS

MODEL_PATH = "models/lgbm_model.pkl"
FEAT_PATH  = "models/feature_cols.pkl"


def predict_next_day(ticker: str = "^GSPC"):
    """
    Fetch latest data, engineer features, and predict
    next trading day's price direction for the given ticker.
    """
    print(f"\n🔍 Running inference for: {ticker}")

    # Load model and feature columns
    model       = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEAT_PATH)

    # Fetch fresh data and engineer features
    raw_df      = fetch_data(ticker=ticker, period="1y")
    feat_df     = engineer_features(raw_df)

    # Take only the last row (most recent trading day)
    latest      = feat_df[feature_cols].iloc[[-1]]
    latest_date = feat_df.index[-1].strftime("%Y-%m-%d")

    # Predict
    pred  = model.predict(latest)[0]
    prob  = model.predict_proba(latest)[0]

    direction   = "UP ↑" if pred == 1 else "DOWN ↓"
    confidence  = prob[1] if pred == 1 else prob[0]

    print(f"\n📅 Latest Trading Date Used : {latest_date}")
    print(f"🔮 Predicted Next-Day Direction : {direction}")
    print(f"📊 Confidence                  : {confidence:.2%}")
    print("\n📌 Feature Snapshot (latest trading day):")
    print(latest.T.rename(columns={latest.index[0]: "Value"}).to_string())

    return direction, confidence