import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

PRICE_CAP = 1778.6
CYCLES_PER_DAY = 48

def evaluate_and_plot(global_model, X_test, Y_test, Y_base_test, feature_cols, OUTPUT_DIR):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    Y_pred_res_flat = global_model.predict(X_test)
    
    Y_pred_log_flat = np.log1p(Y_base_test) + Y_pred_res_flat
    
    Y_pred_flat = np.clip(np.expm1(Y_pred_log_flat), 0, PRICE_CAP)
    Y_test_actual_flat = Y_test + Y_base_test

    test_rmse = np.sqrt(mean_squared_error(Y_test_actual_flat, Y_pred_flat))
    test_mae  = mean_absolute_error(Y_test_actual_flat, Y_pred_flat)
    print(f'[EVAL] Overall_Test_Metrics: RMSE={test_rmse:.2f} | MAE={test_mae:.2f}')

    print('\n' + '='*50)
    print('FAIR_EVALUATION (OUTLIERS_EXCLUDED)')
    print('='*50)
    
    mask_outliers = (Y_test_actual_flat > 100) & (Y_test_actual_flat < 1778.0)
    y_true_clean = Y_test_actual_flat[mask_outliers]
    y_pred_clean = Y_pred_flat[mask_outliers]
    
    if len(y_true_clean) > 0:
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse_clean = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        print(f'[EVAL] Valid_samples: {len(y_true_clean)} / {Y_test_actual_flat.size} ({(len(y_true_clean)/Y_test_actual_flat.size)*100:.1f}%)')
        print(f'[EVAL] MAE_Clean: {mae_clean:.2f} VNĐ')
        print(f'[EVAL] RMSE_Clean: {rmse_clean:.2f} VNĐ')
        print(f'[EVAL] MAPE_Clean: {mape:.2f}%')
    else:
        print("[WARN] No_samples_in_fair_evaluation_range.")
    print('='*50 + '\n')

    gain = global_model.feature_importance(importance_type='gain')
    ranking = pd.DataFrame({'feature': feature_cols, 'avg_gain': gain})
    ranking = ranking.sort_values('avg_gain', ascending=False).reset_index(drop=True)
    
    top = ranking.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(20), top['avg_gain'].values[::-1], color='steelblue', alpha=0.8)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=9)
    ax.set_title('Top 20 Biến Đặc Trưng (Gain)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    residuals = Y_test_actual_flat - Y_pred_flat
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].hist(residuals, bins=100, color='purple', alpha=0.7)
    axes[0].set_title('Phân phối sai số (Thực tế - Dự báo)')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(Y_test_actual_flat, Y_pred_flat, 'o', color='purple', alpha=0.3, ms=2)
    axes[1].plot([0, PRICE_CAP], [0, PRICE_CAP], 'r--', lw=2)
    axes[1].set(xlabel='Thực Tế', ylabel='Dự Báo', title='Biểu đồ Scatter (Thực tế vs Dự báo)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

