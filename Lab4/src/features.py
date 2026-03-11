"""
features.py
-----------
Loads S&P 500 historical OHLCV data from a bundled CSV file
and engineers technical indicators used as model features.

Indicators engineered:
  - Simple Moving Averages (SMA_10, SMA_30)
  - Relative Strength Index (RSI_14)
  - Daily Return and Volatility (rolling std of returns)
  - Volume Change (% change in volume)
  - Momentum (close price change over 5 days)
  - SMA Ratio (SMA_10 / SMA_30)
  - Target: 1 if next day close > today close, else 0
"""

import pandas as pd
import numpy as np

CSV_PATH = "SP500.csv"


def fetch_data(ticker: str = "^GSPC", period: str = "5y") -> pd.DataFrame:
    """Load S&P 500 OHLCV data from bundled CSV."""
    print(f"📥 Loading S&P 500 data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, header=[0, 1], index_col=0, parse_dates=True)

    # Flatten MultiIndex columns if present (yfinance CSV format)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.dropna(inplace=True)
    print(f"✅ Loaded {len(df)} rows of data.")
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator columns to the dataframe and
    create binary target: 1 = price up next day, 0 = price down.
    """
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df = df.copy()
    df["SMA_10"]       = close.rolling(window=10).mean()
    df["SMA_30"]       = close.rolling(window=30).mean()
    df["RSI_14"]       = compute_rsi(close, window=14)
    df["Daily_Return"] = close.pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(window=10).std()
    df["Volume_Change"]= volume.pct_change()
    df["Momentum_5"]   = close - close.shift(5)
    df["SMA_Ratio"]    = df["SMA_10"] / (df["SMA_30"] + 1e-9)

    # Target: did price go UP the next trading day?
    df["Target"] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)

    print(f"✅ Feature engineering complete. Dataset shape: {df.shape}")
    return df


FEATURE_COLS = [
    "SMA_10", "SMA_30", "RSI_14",
    "Daily_Return", "Volatility",
    "Volume_Change", "Momentum_5", "SMA_Ratio"
]
TARGET_COL = "Target"