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
        # 1. Feature Importance (from LGBM base model)
        plt.figure(figsize=(12, 8))
        if hasattr(model, 'models') and 'lgb' in model.models:
            importances = pd.Series(model.models['lgb'].feature_importances_, index=feature_cols)
        else:
            importances = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols)
            
        importances = importances.sort_values(ascending=False).head(20)
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

        # 4. Error Distribution Histogram (clean regime)
        errors_clean = y_pred_clean - y_true_clean
        plt.figure(figsize=(10, 6))
        plt.hist(errors_clean, bins=60, color='#27ae60', edgecolor='white', alpha=0.85)
        plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
        plt.title('Prediction Error Distribution (Clean Regime)', fontsize=14, pad=15)
        plt.xlabel('Error (Predicted - Actual) VND', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'error_distribution.png', dpi=300)
        plt.close()

        # 5. MAE by Cycle (hour of day)
        df_eval = pd.DataFrame({'actual': Y_test, 'pred': Y_pred}, index=df_test.index)
        df_eval['cycle'] = df_eval.index.hour * 2 + (df_eval.index.minute == 30).astype(int)
        df_eval['abs_error'] = np.abs(df_eval['actual'] - df_eval['pred'])
        mae_by_cycle = df_eval.groupby('cycle')['abs_error'].mean()

        plt.figure(figsize=(14, 5))
        colors = ['#e74c3c' if c > 15 else '#3498db' for c in mae_by_cycle.index]
        plt.bar(mae_by_cycle.index, mae_by_cycle.values, color=colors, edgecolor='white')
        plt.title('MAE by Cycle (blue = morning available, red = blindspot)', fontsize=13, pad=15)
        plt.xlabel('Cycle (0=00:00, 47=23:30)', fontsize=12)
        plt.ylabel('MAE (VND)', fontsize=12)
        plt.xticks(range(0, 48, 4))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'mae_by_cycle.png', dpi=300)
        plt.close()
        
        # 6. Monthly Charts (For every month in test set)
        for (year, month), group in df_test.groupby([df_test.index.year, df_test.index.month]):
            m_mae = mean_absolute_error(group['smp_system_price'], group['pred'])
            m_rmse = np.sqrt(mean_squared_error(group['smp_system_price'], group['pred']))
            m_denom = np.abs(group['smp_system_price']).sum()
            m_wmape = 100 * np.sum(np.abs(group['smp_system_price'] - group['pred'])) / m_denom if m_denom > 0 else np.nan
            
            plt.figure(figsize=(18, 6))
            plt.plot(group.index, group['smp_system_price'], label='Actual SMP', color='#2c3e50')
            plt.plot(group.index, group['pred'], label='Predicted SMP', color='#f39c12')
            plt.title(f'SMP forecast {year}-{month:02d} | MAE={m_mae:.2f}, RMSE={m_rmse:.2f}, WMAPE={m_wmape:.2f}%', fontsize=12)
            plt.ylabel('SMP Price')
            plt.xlabel('Datetime')
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'forecast_{year}_{month:02d}.png', dpi=300)
            plt.close()

        # 6. Write metrics summary
        with open(Path(output_dir) / 'metrics.txt', 'w') as f:
            f.write(f'RMSE (all): {full_rmse:.2f}\n')
            f.write(f'MAE (all): {full_mae:.2f}\n')
            if len(y_true_clean) > 0:
                f.write(f'MAE (clean): {mae_clean:.2f}\n')
                f.write(f'RMSE (clean): {rmse_clean:.2f}\n')
                f.write(f'MAPE (clean): {mape_clean:.2f}%\n')
                f.write(f'WMAPE (clean): {wmape_clean:.2f}%\n')
                f.write(f'Valid samples: {len(y_true_clean)} / {len(Y_test)}\n')
