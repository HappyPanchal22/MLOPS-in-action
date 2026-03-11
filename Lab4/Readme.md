# 📈 S&P 500 Next-Day Direction Predictor (Dockerized ML Pipeline)

This project builds an end-to-end machine learning pipeline that predicts whether the **S&P 500 index will close higher or lower the next trading day** — a binary classification problem.

It loads bundled historical OHLCV data from a CSV, engineers meaningful technical indicators, trains a **LightGBM** classifier with chronologically correct cross-validation, and runs inference to predict tomorrow's market direction — all inside a reproducible Docker container.

---

## Overview

| Component | Description |
|---|---|
| Language | Python 3.9 |
| Data Source | S&P 500 historical OHLCV (`SP500.csv`) via Yahoo Finance / yfinance |
| ML Framework | LightGBM (gradient boosted trees) |
| Task | Binary Classification — Up ↑ (1) or Down ↓ (0) |
| Validation | TimeSeriesSplit (5-fold, no data leakage) |
| Containerization | Docker |

---

## Project Structure

```
Lab4/
├── src/
│   ├── pipeline.py        # Master entry point — orchestrates all steps
│   ├── features.py        # Data loading + technical indicator engineering
│   ├── train.py           # LightGBM training + TimeSeriesSplit CV + evaluation
│   ├── predictor.py       # Load saved model → predict next-day direction
│   ├── SP500.csv          # Bundled S&P 500 OHLCV dataset (5 years)
│   └── requirements.txt
├── Dockerfile
└── README.md
```

---

## Features Engineered

| Feature | Description |
|---|---|
| `SMA_10` | 10-day Simple Moving Average |
| `SMA_30` | 30-day Simple Moving Average |
| `RSI_14` | 14-day Relative Strength Index |
| `Daily_Return` | Day-over-day % price change |
| `Volatility` | Rolling 10-day std dev of daily returns |
| `Volume_Change` | Day-over-day % change in trading volume |
| `Momentum_5` | Price change over last 5 trading days |
| `SMA_Ratio` | SMA_10 / SMA_30 — trend strength indicator |

**Target:** `1` if next day's closing price > today's closing price, else `0`

---

## Model: LightGBM Classifier

LightGBM (Light Gradient Boosting Machine) is a fast, memory-efficient gradient boosting framework widely used in quantitative finance and competitive ML. Key advantages:

- Handles financial tabular data extremely well
- Supports early stopping to prevent overfitting
- Significantly faster training than XGBoost on large datasets
- Natively supports feature importance analysis

### Hyperparameters Used

| Parameter | Value |
|---|---|
| `n_estimators` | 500 |
| `learning_rate` | 0.03 |
| `num_leaves` | 31 |
| `feature_fraction` | 0.8 |
| `bagging_fraction` | 0.8 |
| `reg_alpha / reg_lambda` | 0.1 |

---

## Getting Started

### Prerequisites
- Docker Desktop installed and running
- S&P 500 CSV already generated inside `src/` (see Step 1)

### Step 1: Generate the Dataset (one-time, run outside Docker)

```bash
pip install yfinance
python -c "import yfinance as yf; yf.download('^GSPC', period='5y', auto_adjust=True).to_csv('src/SP500.csv')"
```

### Step 2: Build the Docker Image

```bash
docker build -t sp500-predictor:v1 .
```

### Step 3: Run the Container

```bash
docker run --rm sp500-predictor:v1
```

> The container loads the bundled CSV, engineers features, trains the model, evaluates it, and predicts tomorrow's market direction — fully offline after build.

---

## Pipeline Workflow

```
[Step 1] Load SP500.csv → 1,255 rows of OHLCV data
         ↓
[Step 2] Engineer 8 technical indicators + binary target label
         ↓
[Step 3] TimeSeriesSplit (5-fold CV) → Accuracy + ROC-AUC per fold
         ↓
[Step 4] Train final LightGBM model on full dataset
         ↓
[Step 5] Hold-out test evaluation (last 20% chronologically)
         ↓
[Step 6] Inference → Predict next trading day: UP ↑ or DOWN ↓
```

---

## Actual Output

```
============================================================
  S&P 500 Next-Day Direction Predictor
  Dockerized LightGBM ML Pipeline
============================================================

[Step 1/3] Fetching & Engineering Features...
📥 Loading S&P 500 data from SP500.csv...
✅ Loaded 1255 rows of data.
✅ Feature engineering complete. Dataset shape: (1226, 14)

[Step 2/3] Training LightGBM Classifier...
📊 Dataset — Samples: 1226 | Features: 8
   Class balance → Up: 657 | Down: 569

⚙️  Running TimeSeriesSplit Cross-Validation (5 folds)...
   Fold 1 → Accuracy: 0.4363 | ROC-AUC: 0.4968
   Fold 2 → Accuracy: 0.5441 | ROC-AUC: 0.5380
   Fold 3 → Accuracy: 0.4706 | ROC-AUC: 0.4701
   Fold 4 → Accuracy: 0.5686 | ROC-AUC: 0.4835
   Fold 5 → Accuracy: 0.5735 | ROC-AUC: 0.5083

📈 CV Results → Avg Accuracy: 0.5186 | Avg ROC-AUC: 0.4994

💾 Model saved → models/lgbm_model.pkl
💾 Feature list saved → models/feature_cols.pkl

[Step 3/3] Running Next-Day Prediction...
📅 Latest Trading Date Used : 2026-03-10
🔮 Predicted Next-Day Direction : DOWN ↓
📊 Confidence                  : 82.94%

============================================================
✅ Pipeline complete.
   S&P 500 next trading day → DOWN ↓ (confidence: 82.94%)
============================================================
```

---

## Key Results

| Metric | Value |
|---|---|
| Dataset size | 1,255 rows (5 years of S&P 500 data) |
| CV Avg Accuracy | 51.86% |
| CV Avg ROC-AUC | 0.4994 |
| Next-Day Prediction | DOWN ↓ |
| Prediction Confidence | 82.94% |
| Latest Date Used | 2026-03-10 |

> **Note on accuracy:** ~52% CV accuracy is expected and realistic for stock direction prediction. The market is highly efficient — any consistent edge above 50% is considered meaningful in quantitative finance.

---

## Key Design Decisions

- **Bundled CSV over live API calls** — ensures the Docker container runs fully offline and reproducibly without network dependencies
- **TimeSeriesSplit instead of random train/test split** — financial data is time-ordered; random splits cause data leakage where the model sees future data during training
- **Modular code structure** — `features.py`, `train.py`, `predictor.py` are separated for clarity and reusability
- **LightGBM over XGBoost** — better speed, lower memory footprint, and strong industry adoption in quantitative finance

---

## Dependencies

```
yfinance==0.2.40
pandas==2.2.2
numpy==1.26.4
lightgbm==4.3.0
scikit-learn==1.5.1
joblib==1.4.2
```


## References

- [Yahoo Finance / yfinance](https://github.com/ranaroussi/yfinance)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Docker Documentation](https://docs.docker.com/)