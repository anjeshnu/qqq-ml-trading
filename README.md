# QQQ Directional Forecasting — SVM-Based Trading Strategy

> **Predicting short-term market direction using a multi-stage feature selection pipeline and Support Vector Machine classifier on the Nasdaq-100 ETF (QQQ), with a full backtest and performance attribution.**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Repository Structure](#repository-structure)
5. [Data](#data)
6. [Getting Started](#getting-started)
7. [Dependencies](#dependencies)
8. [Key Findings](#key-findings)

---

## Project Overview

This project builds a **binary classification system** to predict whether the QQQ ETF (Nasdaq-100 tracker) will return more than **+0.5%** on the next trading day. A long-only signal is then used to run a backtest comparing the strategy against a passive buy-and-hold benchmark over the 2015–2025 period.

The project demonstrates:
- **Feature engineering** across multiple asset classes (equities, bonds, volatility, commodities)
- **Three-stage feature selection funnel** (Filter → Wrapper → Embedded)
- **SVM model tuning** with time-series cross-validation
- **Rigorous out-of-sample evaluation** and performance attribution via `pyfolio`

---

## Methodology

### 1. Data Preparation & Feature Engineering

Six datasets covering **2005–2025** were combined into a unified feature matrix:

| Asset | Role |
|-------|------|
| **QQQ** | Target asset (Nasdaq-100 ETF) |
| **S&P 500** | Broad market benchmark |
| **VIX** | Equity volatility / fear gauge |
| **TLT** | Long-duration Treasury bond ETF (rates proxy) |
| **Crude Oil** | Energy / growth inflation proxy |
| **Gold** | Safe-haven / inflation hedge |

For **QQQ**, a rich set of technical features was computed:

- **Returns & Momentum** — 1-day, 5-day, log returns
- **Moving Averages** — EMA/SMA (5 vs. 20 days) and their ratios
- **Volatility** — Rolling std (10-day, 20-day), ATR (5-day, 20-day)
- **Oscillators** — RSI (5, 20), Bollinger Bands (5, 20)
- **Volume** — 20-day z-score

For cross-assets (VIX, TLT, Gold, Oil, SPX), level, return and log-return features were computed. Three **composite macro signals** were also engineered:

| Composite Feature | Formula | Interpretation |
|-------------------|---------|----------------|
| Relative Strength (Nasdaq vs. S&P) | `qqq_ret_5 − spx_ret_5` | Tech sector outperformance |
| Risk Aversion Proxy | `vix_logret_1 − tlt_logret_1` | Investor risk-off sentiment |
| Inflation Beta | `crude_ret_5 − gold_ret_5` | Growth vs. defensive inflation hedge |

**Binary target:** `y = 1` if next-day QQQ return > 0.5%, else `0`.

---

### 2. Feature Selection — Three-Stage Funnel

A sequential funnel was used to reduce ~70 raw features down to a compact, interpretable final set while controlling for redundancy, overfitting, and temporal leakage.

```
All Features (~70)
       │
       ▼
 [Stage 1] Filter — Mutual Information
       │  Keeps top-40 features by non-linear statistical dependency
       │
       ▼
 [Stage 2] Wrapper — Recursive Feature Elimination (RFE)
       │  Logistic Regression + iterative elimination → 25 features
       │
       ▼
 [Stage 3] Embedded — L1 Regularization (Lasso-LR)
       │  LogisticRegressionCV + TimeSeriesSplit + ROC-AUC → 11 features
       │
       ▼
  Final Feature Set (11)
```

| Stage | Method | Features Kept | Key Rationale |
|-------|--------|--------------|---------------|
| Filter | Mutual Information | 40 | Captures non-linear relevance |
| Wrapper | RFE (Logistic Regression) | 25 | Removes redundancy under linearity |
| Embedded | L1 (`LogisticRegressionCV`, TSCV) | **11** | Enforces sparsity, respects time-series structure |

**Final 11 features:**
`vix_level`, `snp500_ret_1`, `qqq_rsi_20`, `risk_aversion_proxy`, `tlt_ret_5`, `qqq_vol_20`, `qqq_vol_z`, `qqq_logret_1`, `qqq_lb_5`, `qqq_volume`, `qqq_atr_20`

---

### 3. SVM Model — Training & Tuning

**Classifier:** `SVC` with `probability=True`, `class_weight='balanced'`

**Pipeline:**
```
SimpleImputer (median) → StandardScaler → SVC
```

**Hyperparameter grid** searched via `GridSearchCV` with 5-fold `TimeSeriesSplit`:

| Hyperparameter | Values Searched |
|---------------|----------------|
| `kernel` | `rbf`, `poly` |
| `C` | `0.5, 1, 2, 5, 10` |
| `gamma` | `scale, 0.1, 0.05, 0.01` |
| `degree` (poly) | `2, 3` |

**Best parameters:** `kernel=rbf`, `C=0.5`, `gamma=0.01`, `degree=2`

---

## Results

### Model Performance (Hold-out Test Set — last 20%)

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.6129** |
| Accuracy | 0.5419 |
| Precision | 0.3861 |
| Recall | 0.6595 |
| F1-Score | 0.487 |

A ROC-AUC of **0.61** is consistent with the academic literature on daily equity direction prediction, where the signal-to-noise ratio is inherently low.

---

### Backtest Performance (2015–2025)

The SVM signal was used to build a long-only strategy: enter QQQ when the model predicts `y=1`, hold cash otherwise.

| Metric | Full Period | In-Sample | Out-of-Sample |
|--------|------------|-----------|---------------|
| **Annual Return** | 8.69% | 5.07% | **24.43%** |
| **Cumulative Return** | 152.7% | 55.4% | 62.7% |
| **Sharpe Ratio** | 0.53 | 0.35 | **1.33** |
| **Sortino Ratio** | 0.74 | 0.48 | 2.01 |
| **Max Drawdown** | -37.2% | -37.2% | -23.8% |
| **Calmar Ratio** | 0.23 | 0.14 | 1.03 |
| Beta (vs QQQ) | 0.78 | 0.78 | 0.78 |
| Alpha | -0.03 | -0.05 | +0.06 |

**Key takeaway:** The strategy underperforms passive buy-and-hold on cumulative return but delivers meaningfully lower volatility and drawdowns. Its strongest utility is as a **risk-management or exposure-scaling overlay**, particularly during volatility spikes — not as a standalone alpha generator.

---

## Repository Structure

```
qqq-ml-trading/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── qqq_svm_trading_strategy.ipynb   ← Main analysis notebook
│
├── data/
│   ├── raw/                             ← Source OHLCV files (2005–2025)
│   │   ├── QQQ_raw.csv
│   │   ├── SNP500_raw.csv
│   │   ├── VIX_raw.csv
│   │   ├── TLT_raw.csv
│   │   ├── Gold_raw.csv
│   │   └── CrudeOil_raw.csv
│   │
│   └── processed/
│       └── combined_dataset_patched.csv ← Feature-engineered dataset
│
└── images/                              ← Chart outputs (populated on run)
```

---

## Data

All raw data consists of daily **OHLCV** (Open, High, Low, Close, Volume) prices aligned to a common business-day calendar.

| File | Asset | Period |
|------|-------|--------|
| `QQQ_raw.csv` | Nasdaq-100 ETF | 2005–2025 |
| `SNP500_raw.csv` | S&P 500 Index | 2005–2025 |
| `VIX_raw.csv` | CBOE Volatility Index | 2005–2025 |
| `TLT_raw.csv` | iShares 20+ Year Treasury ETF | 2005–2025 |
| `Gold_raw.csv` | Gold Spot / ETF | 2005–2025 |
| `CrudeOil_raw.csv` | WTI Crude Oil | 2005–2025 |

The processed dataset (`combined_dataset_patched.csv`) contains all engineered features and is used directly by the modeling sections of the notebook.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/qqq-ml-trading.git
cd qqq-ml-trading
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/qqq_svm_trading_strategy.ipynb
```

The notebook is fully self-contained and runs end-to-end:
- **Data Preparation** (cells 1–10): loads raw CSVs, engineers features, saves processed dataset
- **Part B — Feature Selection** (cells 11–26): runs the three-stage funnel
- **Part C — SVM Modelling** (cells 27–37): trains, tunes, and evaluates the classifier
- **Trading Strategy** (cells 38–42): runs the backtest and generates `pyfolio` tear sheets

> **Note:** If the processed dataset already exists at `data/processed/combined_dataset_patched.csv`, the Data Preparation section can be skipped.

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
pyfolio
tabulate
```

See `requirements.txt` for pinned versions.

---

## Key Findings

1. **Short-term QQQ direction is weakly but consistently predictable** — ROC-AUC of 0.61 aligns with the efficient market hypothesis: signals exist but are small and regime-dependent.

2. **Volatility and risk-sentiment features dominate predictive power** — `vix_level`, `qqq_vol_20`, and `risk_aversion_proxy` were the most important features across both the L1 selection and permutation importance analysis.

3. **RBF kernel outperforms linear SVM** — the nonlinear decision boundary captures conditional relationships between volatility, momentum, and cross-asset signals that a linear model cannot.

4. **The strategy is best used as a risk filter, not an alpha engine** — lower drawdowns (−37% vs. buy-and-hold) and smoother Sharpe in out-of-sample periods suggest value as a tactical overlay within a diversified portfolio.

5. **Out-of-sample Sharpe of 1.33 (2023–2025)** indicates the model adapted well to the post-COVID macro regime, though longer out-of-sample evaluation and rolling retraining would be needed to validate robustness.

---

*This project was completed as part of the Certificate in Quantitative Finance (CQF) programme, Module 4.*
