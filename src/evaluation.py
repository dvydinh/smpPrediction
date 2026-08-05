import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pathlib import Path

def evaluate_and_plot(model, df, feature_cols, output_dir="outputs/kaggle_runs"):
    df_clean = df.dropna(subset=['smp_system_price'] + feature_cols).copy()
    test_mask = df_clean.index.year == 2026
    df_test = df_clean[test_mask].copy()
    
    if len(df_test) == 0:
        print("No test data available for 2026.")
        return
        
    X_test = df_test[feature_cols]
    Y_test = df_test['smp_system_price'].values
    
    Y_pred = model.predict(X_test)
    df_test['pred'] = Y_pred
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # Full Metrics
    full_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    full_mae = mean_absolute_error(Y_test, Y_pred)
    
    print(f"Overall test metrics - RMSE: {full_rmse:.2f} | MAE: {full_mae:.2f}")
    
    # Clean Metrics (exclude outliers: price <= 500 or >= 2500)
    clean_mask = (Y_test > 500) & (Y_test < 2500)
    y_true_clean = Y_test[clean_mask]
    y_pred_clean = Y_pred[clean_mask]
    
    if len(y_true_clean) > 0:
        mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse_clean = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mape_clean = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
        denom = np.abs(y_true_clean).sum()
        wmape_clean = 100 * np.sum(np.abs(y_true_clean - y_pred_clean)) / denom if denom > 0 else np.nan
        
        print(f"Valid samples: {len(y_true_clean)} / {len(Y_test)} ({(len(y_true_clean)/len(Y_test))*100:.1f}%)")
        print(f"MAE (clean): {mae_clean:.2f} VND")
        print(f"RMSE (clean): {rmse_clean:.2f} VND")
        print(f"MAPE (clean): {mape_clean:.2f}%")
        print(f"WMAPE (clean): {wmape_clean:.2f}%")
    
    # Visualizations
    if output_dir:
        # 1. Feature Importance
        plt.figure(figsize=(12, 8))
        importances = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols).sort_values(ascending=False).head(20)
        importances.sort_values().plot(kind='barh', color='#3498db')
        plt.title('Top 20 Feature Importances (Gain)', fontsize=14, pad=15)
        plt.xlabel('Gain', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'feature_importance.png', dpi=300)
        plt.close()
        
        # 2. Time-series Sample (1 week)
        sample_df = df_test.iloc[48*7 : 48*14] # Week 2 of test set
        plt.figure(figsize=(15, 6))
        plt.plot(sample_df.index, sample_df['smp_system_price'], label='Actual SMP', color='#2c3e50', linewidth=2)
        plt.plot(sample_df.index, sample_df['pred'], label='Predicted SMP', color='#e74c3c', linewidth=2, linestyle='--')
        plt.title('SMP Forecast vs Actual (1 Week Sample)', fontsize=14, pad=15)
        plt.ylabel('Price (VND/kWh)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'forecast_vs_actual_sample.png', dpi=300)
        plt.close()
        
        # 3. Scatter Plot (Clean regime)
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true_clean, y_pred_clean, alpha=0.3, s=15, color='#8e44ad')
        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2)
        plt.title('Actual vs Predicted (Clean Regime)', fontsize=14, pad=15)
        plt.xlabel('Actual SMP', fontsize=12)
        plt.ylabel('Predicted SMP', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'scatter_clean_regime.png', dpi=300)
        plt.close()
