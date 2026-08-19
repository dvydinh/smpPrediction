import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pathlib import Path


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def evaluate_and_plot(model, df, feature_cols, output_dir="outputs/kaggle_runs"):
    df_clean = df.dropna(subset=['smp_system_price'] + feature_cols).copy()
    test_mask = df_clean.index.year == 2026
    df_test = df_clean[test_mask].copy()
    
    if len(df_test) == 0:
        print("No test data available for 2026.")
        return
        
    X_test = df_test[feature_cols]
    Y_test = df_test['smp_system_price'].values
    
    if hasattr(model, 'predict_walk_forward'):
        history_actual = df.loc[
            df.index < df_test.index.min(),
            'smp_system_price',
        ].dropna()
        Y_pred = model.predict_walk_forward(
            X_test,
            Y_test,
            history_actual=history_actual,
        )
    else:
        Y_pred = model.predict(X_test)
    Y_pred = np.clip(Y_pred, 0.0, 1778.6)
    df_test['pred'] = Y_pred
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df_test[['smp_system_price', 'pred']].to_csv(
            Path(output_dir) / 'predictions_2026.csv',
            index_label='datetime',
        )
        
    # Full Metrics
    full_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    full_mae = mean_absolute_error(Y_test, Y_pred)
    full_denom = np.abs(Y_test).sum()
    full_wmape = 100 * np.sum(np.abs(Y_test - Y_pred)) / full_denom if full_denom > 0 else np.nan

    print(f"Overall test metrics - RMSE: {full_rmse:.2f} | MAE: {full_mae:.2f} | WMAPE: {full_wmape:.2f}%")

    collapse_mask = Y_test <= 500.0
    regime_probability = None
    regime_weight = None
    if hasattr(model, 'models') and 'regime' in model.models:
        if hasattr(model, 'predict_gate_probability'):
            regime_probability = model.predict_gate_probability(X_test)
        else:
            regime_probability, _, _ = model.models['regime'].predict_components(X_test)
        regime_weight = model._gate_weight(regime_probability)
        predicted_collapse = regime_weight >= 0.5
    else:
        predicted_collapse = Y_pred <= 500.0
    collapse_mae = mean_absolute_error(
        Y_test[collapse_mask],
        Y_pred[collapse_mask],
    ) if collapse_mask.any() else np.nan
    true_positive = np.sum(collapse_mask & predicted_collapse)
    collapse_precision = (
        true_positive / predicted_collapse.sum()
        if predicted_collapse.sum() else 0.0
    )
    collapse_recall = true_positive / collapse_mask.sum() if collapse_mask.sum() else 0.0
    print(
        f"Collapse regime - MAE: {collapse_mae:.2f} | "
        f"Precision: {collapse_precision:.3f} | Recall: {collapse_recall:.3f}"
    )
    if output_dir and regime_probability is not None:
        pd.DataFrame({
            'actual': Y_test,
            'predicted': Y_pred,
            'collapse_probability': regime_probability,
            'collapse_weight': regime_weight,
            'actual_collapse': collapse_mask.astype(int),
            'predicted_collapse': predicted_collapse.astype(int),
        }, index=df_test.index).to_csv(
            Path(output_dir) / 'regime_events.csv',
            index_label='datetime',
        )
    
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
        target_met = wmape_clean < 10.0 and mae_clean < 150.0
        print(f"Target met: {target_met}")

        if 'smp_same_cycle_1d' in df_test.columns:
            baseline = np.clip(df_test['smp_same_cycle_1d'].to_numpy(), 0.0, 1778.6)
            baseline_clean = baseline[clean_mask]
            baseline_mae = mean_absolute_error(y_true_clean, baseline_clean)
            baseline_wmape = 100 * np.sum(np.abs(y_true_clean - baseline_clean)) / denom
            print(f"Seasonal baseline - MAE: {baseline_mae:.2f} | WMAPE: {baseline_wmape:.2f}%")
    
    # Visualizations
    if output_dir:
        # 1. Feature Importance (from LGBM base model)
        plt.figure(figsize=(12, 8))
        if hasattr(model, 'models') and 'lgb' in model.models:
            importances = pd.Series(model.models['lgb'].feature_importances_, index=feature_cols)
        else:
            importances = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols)
        
        # Print ALL feature importances for analysis
        importances_sorted = importances.sort_values(ascending=False)
        print("\n=== FULL FEATURE IMPORTANCE (Gain) ===")
        for rank, (feat, score) in enumerate(importances_sorted.items(), 1):
            print(f"  {rank:3d}. {feat:40s} {score:12.1f}")
        print(f"  Total features: {len(importances_sorted)}")
        print("=" * 50)
            
        importances = importances_sorted.head(20)
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

        cycle_rows = []
        for cycle, group in df_eval.groupby('cycle'):
            group_clean = group[(group['actual'] > 500.0) & (group['actual'] < 2500.0)]
            clean_denom = np.abs(group_clean['actual']).sum()
            cycle_rows.append({
                'cycle': int(cycle),
                'samples': int(len(group)),
                'mae': float(group['abs_error'].mean()),
                'clean_samples': int(len(group_clean)),
                'clean_mae': float(group_clean['abs_error'].mean()),
                'clean_wmape': float(
                    100.0 * group_clean['abs_error'].sum() / clean_denom
                    if clean_denom > 0 else np.nan
                ),
            })
        pd.DataFrame(cycle_rows).to_csv(
            Path(output_dir) / 'metrics_by_cycle.csv',
            index=False,
        )

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
        monthly_rows = []
        for (year, month), group in df_test.groupby([df_test.index.year, df_test.index.month]):
            m_mae = mean_absolute_error(group['smp_system_price'], group['pred'])
            m_rmse = np.sqrt(mean_squared_error(group['smp_system_price'], group['pred']))
            m_denom = np.abs(group['smp_system_price']).sum()
            m_wmape = 100 * np.sum(np.abs(group['smp_system_price'] - group['pred'])) / m_denom if m_denom > 0 else np.nan
            group_clean = group[
                (group['smp_system_price'] > 500.0)
                & (group['smp_system_price'] < 2500.0)
            ]
            clean_error = np.abs(group_clean['smp_system_price'] - group_clean['pred'])
            clean_denom = np.abs(group_clean['smp_system_price']).sum()
            monthly_rows.append({
                'year': int(year),
                'month': int(month),
                'samples': int(len(group)),
                'mae': float(m_mae),
                'wmape': float(m_wmape),
                'clean_samples': int(len(group_clean)),
                'clean_mae': float(clean_error.mean()),
                'clean_wmape': float(
                    100.0 * clean_error.sum() / clean_denom
                    if clean_denom > 0 else np.nan
                ),
            })
            
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
        pd.DataFrame(monthly_rows).to_csv(
            Path(output_dir) / 'metrics_by_month.csv',
            index=False,
        )

        # 6. Write metrics summary
        with open(Path(output_dir) / 'metrics.txt', 'w') as f:
            f.write(f'RMSE (all): {full_rmse:.2f}\n')
            f.write(f'MAE (all): {full_mae:.2f}\n')
            f.write(f'WMAPE (all): {full_wmape:.2f}%\n')
            if len(y_true_clean) > 0:
                f.write(f'MAE (clean): {mae_clean:.2f}\n')
                f.write(f'RMSE (clean): {rmse_clean:.2f}\n')
                f.write(f'MAPE (clean): {mape_clean:.2f}%\n')
                f.write(f'WMAPE (clean): {wmape_clean:.2f}%\n')
                f.write(f'Valid samples: {len(y_true_clean)} / {len(Y_test)}\n')
                f.write(f'Target met: {target_met}\n')
                if 'smp_same_cycle_1d' in df_test.columns:
                    f.write(f'Baseline MAE (clean): {baseline_mae:.2f}\n')
                    f.write(f'Baseline WMAPE (clean): {baseline_wmape:.2f}%\n')
            f.write(f'Collapse MAE: {collapse_mae:.2f}\n')
            f.write(f'Collapse precision: {collapse_precision:.4f}\n')
            f.write(f'Collapse recall: {collapse_recall:.4f}\n')

        manifest = {
            'selected_feature_count': len(feature_cols),
            'base_models': getattr(model, 'model_order_', []),
            'selected_candidate': getattr(model, 'selected_candidate_', None),
            'online_bias_days': getattr(model, 'bias_days_', 0),
            'online_cap_days': getattr(model, 'cap_days_', 0),
            'online_cap_ratio': getattr(model, 'cap_ratio_', None),
            'meta_kind': getattr(model, 'meta_kind_', 'global'),
            'meta_alpha': float(getattr(getattr(model, 'meta_learner', None), 'alpha_', np.nan)),
            'meta_coefficients': {
                name: float(weight)
                for name, weight in zip(
                    getattr(model, 'model_order_', []),
                    getattr(getattr(model, 'meta_learner', None), 'coef_', []),
                )
            },
            'cycle_bias': getattr(model, 'cycle_bias_', np.zeros(48)).tolist(),
            'base_selection_scores': getattr(model, 'base_selection_scores_', {}),
            'regime_estimators': getattr(model, 'regime_estimators_', {}),
            'gate_threshold': getattr(model, 'gate_threshold_', None),
            'gate_ramp': getattr(model, 'gate_ramp_', None),
            'gate_validation_scores': getattr(model, 'gate_validation_scores_', {}),
            'shape_guard_enabled': getattr(model, 'shape_guard_enabled_', False),
            'state_projection_enabled': getattr(
                model,
                'state_projection_enabled_',
                False,
            ),
            'lower_projection_cut': getattr(
                model,
                'lower_projection_cut_',
                None,
            ),
            'cap_projection_cut': getattr(
                model,
                'cap_projection_cut_',
                None,
            ),
            'lower_state_value': getattr(model, 'lower_state_value_', None),
            'cap_state_value': getattr(model, 'cap_state_value_', None),
            'state_projection_scores': getattr(
                model,
                'state_projection_scores_',
                {},
            ),
            'normal_delta_bounds': getattr(
                model,
                'normal_delta_bounds_',
                np.empty((0, 2)),
            ).tolist(),
            'gate_delta_bounds': getattr(
                model,
                'gate_delta_bounds_',
                np.empty((0, 2)),
            ).tolist(),
        }
        with open(Path(output_dir) / 'model_manifest.json', 'w') as f:
            json.dump(_json_safe(manifest), f, indent=2)
