# VCGM SMP Prediction System

This repository contains the end-to-end production system for forecasting the **System Marginal Price (SMP)** for the Vietnam Competitive Generation Market (VCGM) on a Day-Ahead basis.

## Project Structure

The project is structured following strict Software Engineering and Machine Learning Research standards, completely avoiding flat scripts and spaghetti code.

```text
smpPrediction/
├── data/
│   └── raw/                    # Original raw datasets (Do not modify)
│       ├── market/             # SMP prices, system load, dispatch capacity
│       ├── hydro/              # 66 per-reservoir hourly hydro datasets
│       └── exogenous/          # Weather, Fuel/Macro, and Vietnamese Calendar
│
├── vcgm/                       # Core Python Package
│   ├── __init__.py
│   ├── config.py               # Single source of truth (paths, hyperparams, rules)
│   ├── data/                   # Data ingestion, alignment, and profiling
│   │   ├── loader.py           # Returns clean DatetimeIndexed DataFrames
│   │   ├── alignment.py        # Master 30-min timeline generation
│   │   └── profiler.py         # Missing value auditing and gap detection
│   ├── features/               # Feature Engineering (Zero data-leakage)
│   │   ├── temporal.py         # Cyclical encodings, calendar flags
│   │   ├── lagged.py           # SMP & Load lags, rolling statistics
│   │   ├── exogenous.py        # Hydro aggregation, weather/fuel broadcast
│   │   └── pipeline.py         # Orchestrates the final Design Matrix
│   ├── models/                 # Modeling and Evaluation
│   │   ├── baseline.py         # Naive lag-7 baseline
│   │   ├── direct_forecast.py  # 48-model LightGBM direct forecasting
│   │   └── evaluation.py       # Walk-forward CV and residual analysis
│   └── inference/              # Production Serving
│       └── predictor.py        # predict_tomorrow_smp() with strict failsafes
│
├── scripts/                    # CLI Entrypoints (Thin wrappers)
│   ├── profile_data.py         # Run data quality audit
│   ├── build_features.py       # Construct design_matrix.parquet
│   ├── train.py                # Train the 48 models
│   ├── evaluate.py             # Run Walk-Forward cross-validation
│   └── predict.py              # Generate Day D+1 predictions
│
└── outputs/                    # Generated artifacts (gitignored)
    ├── figures/                # Visualizations
    │   └── eda/                # Exploratory Data Analysis plots
    ├── metrics/                # JSON metrics (RMSE, MAE)
    ├── models/                 # Saved LightGBM .txt files and metadata
    └── processed/              # design_matrix.parquet
```

## Methodology

### 1. Data Integrity & Alignment
We use a **Master Timeline** approach. A continuous 30-minute DatetimeIndex is generated from `2021-01-01` to `2026-06-19`. All heterogeneous data sources (hourly hydro, daily dispatch, 30-min weather) are left-joined onto this timeline to explicitly expose any missing gaps.

### 2. Feature Engineering (Strict Zero Leakage)
Prediction target: Day D+1 (48 cycles).
Inference Execution: 08:00 AM on Day D.
Therefore, the model can **only** use data available up to 07:30 AM on Day D.

To enforce this, all lagged and rolling features in `vcgm.features.lagged` use a minimum `shift(48)` relative to the prediction target. The design matrix is orchestrated by `vcgm.features.pipeline`.

### 3. Direct Multi-Step Forecasting
Instead of recursive one-step-ahead forecasting (which accumulates error), we use **Direct Forecasting**. We train **48 independent LightGBM models** (`M_00` to `M_47`), one for each 30-minute cycle of the day.

We use the **Huber objective function** because the SMP distribution is heavily zero-inflated (~30% of prices are ≤ 2 VND) and has occasional extreme spikes.

### 4. Production Failsafes
The inference module (`vcgm/inference/predictor.py`) implements aggressive fallback logic:
- If Weather API fails -> uses 7-day historical rolling average.
- If Feature Matrix fails entirely -> uses Naive Baseline (same day last week).
- Strict 15-minute execution deadline monitoring.

## Usage

**1. Build the Design Matrix:**
```bash
python scripts/build_features.py
```

**2. Train the models:**
```bash
python scripts/train.py
```

**3. Evaluate using Walk-Forward CV:**
```bash
python scripts/evaluate.py
```

**4. Run Production Inference:**
```bash
python scripts/predict.py --date 2026-06-18
```
