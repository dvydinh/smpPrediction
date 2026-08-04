import json

with open('train_kaggle.ipynb', 'r', encoding='utf-8') as f: nb = json.load(f)

script_lines = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src_str = ''.join(cell['source'])
        if '%%time' in src_str and 'lgb.train' in src_str:
            # Replace training with loading
            script_lines.append("models = {}\n")
            script_lines.append("for k in range(CYCLES_PER_DAY):\n")
            script_lines.append("    models[k] = lgb.Booster(model_file=f'outputs/kaggle_runs/run_20260804_142254/models/lgb_cycle_{k:02d}.txt')\n")
            script_lines.append("print('Loaded 48 models locally.')\n")
        elif 'cv_splits = [' in src_str:
            # Stop before CV
            break
        elif 'model_dir = ' in src_str:
            # Stop before saving
            continue
        elif 'KAGGLE_INPUT.exists()' in src_str:
            # Local paths
            script_lines.append("DATA_ROOT = Path('data/raw')\n")
            script_lines.append("MARKET_DIR = DATA_ROOT / 'market'\n")
            script_lines.append("HYDRO_DIR  = DATA_ROOT / 'hydro'\n")
            script_lines.append("EXOG_DIR   = DATA_ROOT / 'exogenous'\n")
            script_lines.append("OUTPUT_DIR = Path('outputs')\n")
            script_lines.append("PRICE_CAP = 1778.6\n")
            script_lines.append("NEAR_ZERO = 2.0\n")
            script_lines.append("CYCLES_PER_DAY = 48\n")
            script_lines.append("LAG_SHIFT = 48\n")
        elif 'LGB_PARAMS' in src_str:
            script_lines.extend(cell['source'])
            script_lines.append("\n")
        else:
            if 'import kaggle_secrets' not in src_str:
                script_lines.extend(cell['source'])
                script_lines.append("\n")

script_lines.append("""
mask_outliers = (Y_test_actual > 100) & (Y_test_actual < 1778.0)
y_true_clean = Y_test_actual[mask_outliers]
y_pred_clean = Y_pred[mask_outliers]
mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
print('\\n--- KẾT QUẢ ĐÃ LỌC OUTLIER ---')
print(f'Mẫu hợp lệ: {len(y_true_clean)} / {Y_test_actual.size}')
print(f'MAE (Clean): {mae_clean:.2f} VND')
print(f'MAPE (Clean): {mape:.2f}%')
""")

with open('eval_mape.py', 'w', encoding='utf-8') as f:
    f.writelines(script_lines)
