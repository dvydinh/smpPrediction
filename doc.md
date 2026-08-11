# SMP Day-Ahead Forecasting System Documentation

## 1. System Overview

This document describes the end-to-end architecture, methodologies, and findings for the Vietnam Competitive Generation Market (VCGM) System Marginal Price (SMP) Day-Ahead Forecasting system. The primary operational constraint is the 08:00 AM Day D blindspot: forecasts for all 48 cycles of Day D+1 must be generated using only data available prior to 08:00 AM on Day D.

## 2. Dataset Analysis (EDA)

The system relies on three primary data domains covering the period from 2021-01-01 to 2026-06-19 at 30-minute granularity (48 cycles/day).

### 2.1. Market Data (NSMO)
- **Target Variable**: `smp_system_price` (System Marginal Price).
- **Sub-targets**: `smp_north_price`, `smp_central_price`, `smp_south_price`.
- **System Load**: `load_total_mw`, `load_north_mw`, `load_central_mw`, `load_south_mw`.
- **Dispatch Constraints**: Installed and dispatched capacity limits across thermal, hydro, solar, and wind (e.g., `disp_total_installed_mw`).
- **Characteristics**: Highly non-stationary. Subject to extreme spikes during transmission congestion (500kV interface limits) or unexpected thermal plant outages.
- **Spikes vs. Outliers Principle**: Statistically, price spikes (>1500 VNĐ) appear as outliers, but economically, they are legitimate manifestations of Scarcity Pricing. Unlike standard machine learning pipelines which trim outliers, EPF architectures must rigorously preserve spikes, utilizing robust Tree ensembles and Deep Learning architectures to extrapolate extreme scarcity events without structural collapse.

### 2.2. Hydrological Data
- **Variables**: `inflow_m3s`, `discharge_m3s`, `water_level_m` for Hoa Binh, Son La, and Ialy reservoirs.
- **Characteristics**: Seasonal patterns driven by monsoon cycles. Dictates the baseline generation mix and acts as a dampener for SMP volatility.

### 2.3. Exogenous Data
- **Weather**: Temperature, shortwave radiation, wind speed (Open-Meteo API) across 3 regions.
- **Calendar**: Public holidays, day of week, month, cycle identifiers.

## 3. Data Preprocessing & Feature Engineering

Feature engineering strictly simulates the 08:00 Day D operational blindspot. All chronological leakage is prevented.

### 3.1. Autoregressive Lags (The 08:00 Blindspot)
- For Day D+1 cycles `c <= 15` (00:00 to 07:30): The corresponding cycle `c` on Day D has already occurred. Thus, a lag of 1 day `shift(48)` is extracted.
- For Day D+1 cycles `c > 15` (08:00 to 23:30): The corresponding cycle `c` on Day D has not yet occurred at 08:00. Thus, a lag of 2 days `shift(96)` (representing Day D-1) is extracted.

### 3.2. Aggregated Statistics
- **Morning Aggregates (Day D)**: Mean, Min, Max of SMP and Load for the period 00:00 to 07:30 of Day D. Shifted by 1 day to align with Day D+1 targets.
- **Full Day Aggregates (Day D-1)**: Mean, Min, Max, and low-price probability (Gate < 500) for the most recent fully completed day (Day D-1). Shifted by 2 days.

### 3.3. Proxy Derivations
- **Solar/Wind Generation Proxy**: Derived using formula: `(Weather Variable / Normalization Constant) * Installed Capacity * Efficiency Factor`.
- **Residual Load Proxy**: `Lagged Load - Solar Proxy - Wind Proxy`.
- **Thermal Margin Proxy**: `Total Thermal Capacity - Residual Load`. This estimates the remaining reserve capacity.

## 4. Training Methodology & Architecture

The architecture was upgraded from a standalone LightGBM framework to a hybrid ensemble methodology based on recent EPF (Electricity Price Forecasting) research.

### 4.1. Base Learners (Stage 1)
- **LightGBM**: Optimized for MAE (Mean Absolute Error). Efficient at handling non-linear interactions and categorical features. Parameters: `learning_rate=0.01`, `n_estimators=1500`, `num_leaves=127`.
- **XGBoost**: Configured with `reg:absoluteerror`. Provides strict tree structures highly resistant to outlier perturbations. Parameters: `learning_rate=0.01`, `n_estimators=1500`, `max_depth=8`.
- **CatBoost**: Optimized for high-cardinality cyclic temporal parameters without extensive pre-encoding. Parameters: `learning_rate=0.02`, `iterations=1500`, `depth=8`.
- **MLP (Deep Learning)**: A Multi-Layer Perceptron (`MLPRegressor`) encapsulated in a `StandardScaler` Pipeline. Introduces neural network representation learning to compensate for the fundamental extrapolation weaknesses inherent in pure Tree-based ensembles.

### 4.2. Meta-Learner (Stage 2)
- **Ridge Regression (L2 Penalty)**: Operates as a Soft-Gate. It calculates the final forecast using the Out-Of-Fold (OOF) predictions from the Base Learners. The OOF predictions are generated via `TimeSeriesSplit(n_splits=5)` to maintain chronological integrity and prevent data leakage during meta-learner fitting.

## 5. Trial Results & Tuning Iterations

### Trial 1: Single LightGBM (Baseline)
- **Strategy**: Basic gradient boosting with MAE objective.
- **Results**: WMAPE (clean): 15.96%. MAE (clean): 226.32 VNĐ. Overall MAE: 269.16 VNĐ.
- **Analysis**: The model struggled to differentiate between normal market operations and high-volatility regimes due to single-algorithm variance limits.

### Trial 2: Stacking Ensemble (Base Settings)
- **Strategy**: Integration of LGBM, XGBoost, CatBoost with Ridge Regression.
- **Results**: WMAPE (clean): 14.02%. MAE (clean): 198.71 VNĐ. Overall MAE: 239.42 VNĐ.
- **Analysis**: Substantial error reduction (~12%). The meta-learner successfully distributed weights based on regime stability, resolving the Oracle Hard-Gate dilemma.

### Trial 3: Advanced Proxy Features (EWMA, Momentum, Volatility)
- **Strategy**: Adding Exponential Weighted Moving Averages (EWMA), momentum gradients, and rolling standard deviations to artificially capture market trends in the absence of future declaration data.
- **Results**: WMAPE (clean): 14.11%. MAE (clean): 200.09 VNĐ. Overall MAE: 238.47 VNĐ.
- **Analysis**: The inclusion of derivative proxy features failed to improve performance. This demonstrated the "Information Ceiling" principle: derived autoregressive features cannot substitute the deterministic supply/demand variables (Day-Ahead Load Forecasts and Plant Outages). Excess features caused slight overfitting in the tree structures. The codebase was subsequently reverted to the Trial 2 state.

### Trial 4: Capacity Maximization
- **Strategy**: Expanding `n_estimators`/`iterations` to 1500 and decreasing learning rates (0.01-0.02) to maximize learning capacity prior to early stopping triggers.
- **Results**: WMAPE (clean): 13.91%. MAE (clean): 197.25 VNĐ.
- **Analysis**: Allowing models to learn deeper structural representations at a slower pace resulted in breaking the 14% plateau, achieving the lowest historical error for the blindspot constraint.

### Trial 5: Deep Learning Integration (MLP)
- **Strategy**: Introducing a Multi-Layer Perceptron (Neural Network) with standard scaling to the base learner suite to exploit neural representation learning against the extrapolation failures of Decision Trees.
- **Results**: WMAPE (clean): 15.07%. MAE (clean): 213.69 VNĐ.
- **Analysis**: Thất bại. Mô hình mạng nơ-ron MLP tỏ ra không phù hợp với không gian đặc trưng hiện tại của chúng ta (chủ yếu là Lags và Aggregates mang tính tự hồi quy). Nó làm nhiễu tín hiệu của Meta-Learner, khiến kết quả tổng thể bị kéo lùi. Điều này chứng minh rằng với cấu trúc dữ liệu hiện tại, các mô hình Cây (Trees) đã đạt đến cảnh giới tối đa. Chúng ta đã chính thức **loại bỏ** MLP và quay về với cấu trúc Tối đa hoá dung lượng của Trial 4 làm bản Production cuối cùng.

### Trial 6: Extreme Deep Tuning (Max Capacity v2)
- **Strategy**: Pushing tree-based algorithms to their absolute theoretical limits. `n_estimators`/`iterations` increased to 3000, `learning_rate` dropped to 0.005. Added `colsample_bytree=0.8` (column-level sampling only, no row sampling to preserve temporal integrity). Increased `num_leaves` to 255 for LGBM and `depth` to 10 for XGB/CatBoost. Added `l2_leaf_reg=3` for CatBoost regularization and `min_child_samples=20` for LGBM.
- **Results**: WMAPE (clean): 13.26%. MAE (clean): 187.92 VNĐ. RMSE (clean): 255.99 VNĐ. Overall MAE: 225.44 VNĐ.
- **Analysis**: New all-time record. The combination of slower learning rate and deeper tree structures allowed each base learner to capture finer-grained price dynamics without overfitting. Training time increased to ~52 minutes (3100s) but the 0.65% absolute improvement in WMAPE justified the cost. This is now the **Production baseline**.

### Trial 7: Rolling Volatility & Target Log-Transform
- **Strategy**: Added 8 new features capturing price momentum (`smp_momentum_1d_2d`), North-South spread (`smp_spread_ns_1d`), and rolling volatility (1-day and 7-day standard deviations). Additionally, transformed the target variable using `log1p` during training and `expm1` for prediction to help the model compress extreme spikes.
- **Results**: WMAPE (clean): 35.46%. MAE (clean): 502.70 VNĐ. RMSE (clean): 588.89 VNĐ.
- **Analysis**: Complete disaster. The `log1p` target transformation caused MSLE optimization instead of MAE, making small log-space errors explode when exponentiated back to linear space. This trial was reverted.

### Trial 8: Fuel Prices, Hydro Proxy & ICEEMDAN Signal Decomposition
- **Strategy**: Reverted `log1p` target transform from Trial 7. Unlocked existing macro/fuel data (`coal_proxy_price`, `brent_price`, `gas_proxy_price`) as features with safe D-1 lags. Created a `precip_rolling_30d` proxy for hydro reservoir levels and a `hydro_stress_proxy`. Integrated **ICEEMDAN** (Improved Complete Ensemble EMD with Adaptive Noise) as a 4th base model in the Stacking Ensemble. Implemented strict expanding-window decomposition to ensure zero data leakage.
- **Results**: WMAPE (clean): 12.63%. MAE (clean): 179.01 VNĐ. RMSE (clean): 243.63 VNĐ.
- **Analysis**: NEW RECORD! Breaking the 13% barrier. The combination of fuel/macro signals, hydro proxy, and the powerful ICEEMDAN frequency decomposition gave the meta-learner the edge it needed. This is the new state-of-the-art for our pipeline.

### Trial 9: Extreme Optimization (Huber & Cross-interactions)
- **Strategy**: Attempted to break 10% WMAPE by upgrading the meta-learner from Ridge to HuberRegressor, increasing ICEEMDAN maximum IMFs to 8, and introducing cross-domain interactions (`coal_x_residual_load`).
- **Results**: WMAPE (clean): 14.75%. MAE (clean): 209.14 VNĐ.
- **Analysis**: Performance degraded. HuberRegressor likely discarded too much information by heavily down-weighting outliers (which are critical in power price forecasting). Additionally, 8 IMFs in ICEEMDAN may have overfitted the noise, and cross-domain interactions introduced multicollinearity. Reverting to Trial 8.

### Trial 8.1: Trial 8 Revert with GPU Acceleration
- **Strategy**: Reverted to Trial 8 architecture (Ridge + 6 IMFs) but retained the GPU execution flags (`tree_method='hist', device='cuda'` for XGBoost and `task_type='GPU'` for CatBoost).
- **Results**: WMAPE (clean): 11.94%. MAE (clean): 169.26 VNĐ. RMSE (clean): 231.13 VNĐ.
- **Analysis**: **NEW RECORD!** Running the exact same Trial 8 architecture on GPU yielded significantly better results (dropping from 12.63% to 11.94%). This is likely due to structural differences in how GPU implementations construct trees (e.g., histogram binning differences) serving as a beneficial regularizer.

### Trial 13: Physics Features Integration
- **Strategy**: Kept the exact Trial 8.1 architecture (Ridge meta-learner, full 5-year data) but introduced 2 new physics-informed features: `load_to_rad_ratio` (Load / Solar Radiation) and `load_to_wind_ratio` (Load / Wind Speed) to help tree models better identify grid scarcity conditions.
- **Results**: WMAPE (clean): 11.71%. MAE (clean): 165.96 VNĐ. RMSE (clean): 227.31 VNĐ.
- **Analysis**: **CURRENT RECORD (BEST SOTA)!** Adding the physical ratios allowed the models to slice the data more efficiently when solar/wind generation drops during high load periods, reducing MAE by nearly 4 VNĐ without adding complex model bloat.

### Trial 14: Data-Centric Regime & Holiday Flags
- **Strategy**: Implemented true data-centric solutions without proxying the target. Added `is_post_covid` (year >= 2023) to allow tree algorithms to split logic between historical anomalies and the new normal, while retaining all data volume. Added `is_holiday` incorporating exact Vietnam solar/lunar public holidays (Tet, 30/4, Hung King).
- **Results**: WMAPE (clean): 11.33%. MAE (clean): 160.69 VNĐ. RMSE (clean): 219.66 VNĐ.
- **Analysis**: **NEW ALL-TIME RECORD!** Successfully breached the 165 barrier. The regime flag proved that tree models can automatically handle shifting distributions if explicitly labeled, negating the need to cut 40% of the dataset. Vietnam holiday inclusion efficiently handled extreme demand-drop price crashes. We are ~10 VNĐ away from the final target (150).

## 6. Inference Procedure (Production)

The production inference script (`inference_production.py`) is designed for fully automated daily execution at 08:00 AM.

### Workflow:
1. **Model Loading**: Deserializes `stacking_ensemble.pkl`.
2. **Data Ingestion**: Loads historical CSVs and strictly truncates the dataset at 07:30 Day D.
3. **Weather Fetching**: Connects to the Open-Meteo API to retrieve the actual forecast for Day D+1.
4. **Mocking/Proxy Injection**: 
   - Due to the lack of A0's Day-Ahead Load Forecast, Day D+1 load is proxied via a naive forecast using Day D or Day D-1 values.
   - Dispatch capacities are forward-filled from the last known state.
5. **Prediction Vectorization**: Applies `add_engineered_features` over the constructed 48-cycle future dataframe and runs the Stacking predict function.
6. **Output**: Prints the 48 SMP values to the terminal.
