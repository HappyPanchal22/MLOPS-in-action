"""
pipeline.py
-----------
Master entry point. Orchestrates the full ML pipeline:
  1. Fetch S&P 500 OHLCV data via yfinance
  2. Engineer technical indicators as features
  3. Train LightGBM classifier (with TimeSeriesSplit CV)
  4. Evaluate on hold-out test set
  5. Run inference → predict next trading day's direction
"""

from features import fetch_data, engineer_features
from train import train
from predictor import predict_next_day

TICKER = "^GSPC"   # S&P 500 Index
PERIOD = "5y"      # 5 years of historical data


def main():
    print("=" * 60)
    print("  S&P 500 Next-Day Direction Predictor")
    print("  Dockerized LightGBM ML Pipeline")
    print("=" * 60)

    # ── Step 1: Data Ingestion ───────────────────────────────────
    print("\n[Step 1/3] Fetching & Engineering Features...")
    raw_df  = fetch_data(ticker=TICKER, period=PERIOD)
    feat_df = engineer_features(raw_df)

    # ── Step 2: Training ─────────────────────────────────────────
    print("\n[Step 2/3] Training LightGBM Classifier...")
    train(feat_df)

    # ── Step 3: Inference ────────────────────────────────────────
    print("\n[Step 3/3] Running Next-Day Prediction...")
    direction, confidence = predict_next_day(ticker=TICKER)

    print("\n" + "=" * 60)
    print("✅ Pipeline complete.")
    print(f"   S&P 500 next trading day → {direction} "
          f"(confidence: {confidence:.2%})")
    print("=" * 60)


if __name__ == "__main__":
    main()