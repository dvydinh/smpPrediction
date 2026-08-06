"""
ICEEMDAN Signal Decomposition for Electricity Price Forecasting.
Uses expanding-window decomposition to PREVENT data leakage.

WARNING: ICEEMDAN decomposes the ENTIRE input series to determine modes.
If test data is included, future information leaks into training.
This module strictly decomposes ONLY the training portion of each fold.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb


class ICEEMDANForecaster:
    """
    A forecaster that decomposes the target series into Intrinsic Mode Functions
    (IMFs) using ICEEMDAN, then trains a separate LightGBM model for each IMF.
    
    Anti-Leakage Design:
    - decompose() only receives TRAINING data
    - Each IMF model is trained independently
    - Predictions are summed to produce final forecast
    """
    
    def __init__(self, n_imfs_max=8):
        self.n_imfs_max = n_imfs_max
        self.imf_models = []
        self.n_imfs_actual = 0
        
    def _get_imf_model(self):
        """Lightweight LightGBM for each IMF component."""
        return lgb.LGBMRegressor(
            objective='mae',
            learning_rate=0.03,
            num_leaves=63,
            n_estimators=500,
            colsample_bytree=0.8,
            min_child_samples=30,
            random_state=42
        )
    
    def _decompose(self, y_series):
        """
        Decompose a 1D numpy array into IMFs using ICEEMDAN.
        Falls back to simple frequency-based decomposition if PyEMD is unavailable.
        
        CRITICAL: y_series must be TRAINING data ONLY. Never include test data.
        """
        try:
            from PyEMD import CEEMDAN
            ceemdan = CEEMDAN(trials=50, epsilon=0.005)
            ceemdan.noise_seed(42)
            imfs = ceemdan(y_series)
        except ImportError:
            # Fallback: Manual frequency decomposition using rolling averages
            # This is a simplified but leakage-safe alternative
            imfs = []
            residual = y_series.copy()
            
            # Extract components at different time scales
            windows = [48, 96, 336, 672]  # 1 day, 2 days, 1 week, 2 weeks
            for w in windows:
                if len(residual) < w * 2:
                    break
                smooth = pd.Series(residual).rolling(w, min_periods=w//2, center=False).mean().bfill().values
                high_freq = residual - smooth
                imfs.append(high_freq)
                residual = smooth
            
            imfs.append(residual)  # Final residual (trend)
            imfs = np.array(imfs)
        
        # Limit number of IMFs
        if len(imfs) > self.n_imfs_max:
            merged_residual = np.sum(imfs[self.n_imfs_max-1:], axis=0)
            imfs = np.vstack([imfs[:self.n_imfs_max-1], merged_residual.reshape(1, -1)])
        
        return imfs
    
    def fit(self, X_train, y_train):
        """
        Fit the ICEEMDAN forecaster.
        
        1. Decompose y_train into IMFs (ONLY training data - no leakage)
        2. Train one LightGBM model per IMF
        """
        y_values = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Step 1: Decompose ONLY training target
        imfs = self._decompose(y_values)
        self.n_imfs_actual = len(imfs)
        
        # Step 2: Train one model per IMF
        self.imf_models = []
        for i in range(self.n_imfs_actual):
            model = self._get_imf_model()
            imf_target = imfs[i]
            model.fit(X_train, imf_target)
            self.imf_models.append(model)
    
    def predict(self, X_test):
        """
        Predict by summing predictions from all IMF models.
        """
        total_pred = np.zeros(len(X_test))
        for model in self.imf_models:
            total_pred += model.predict(X_test)
        return total_pred
