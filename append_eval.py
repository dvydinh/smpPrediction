import json

with open('train_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "# --- FAIR EVALUATION: EXCLUDING OUTLIERS ---\n",
        "# The market contains extreme outliers (price cap at 1778.6 and near-zero prices).\n",
        "# We evaluate the MAPE and MAE strictly on normal market conditions (prices between 100 and 1778).\n",
        "\n",
        "print('\\n' + '='*50)\n",
        "print('FINAL FAIR EVALUATION (EXCLUDING OUTLIERS)')\n",
        "print('='*50)\n",
        "\n",
        "mask_outliers = (Y_test_actual > 100) & (Y_test_actual < 1778.0)\n",
        "y_true_clean = Y_test_actual[mask_outliers]\n",
        "y_pred_clean = Y_pred[mask_outliers]\n",
        "\n",
        "mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100\n",
        "mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)\n",
        "rmse_clean = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))\n",
        "\n",
        "print(f'Valid evaluation samples: {len(y_true_clean)} / {Y_test_actual.size} ({(len(y_true_clean)/Y_test_actual.size)*100:.1f}%)')\n",
        "print(f'MAE (Clean): {mae_clean:.2f} VND')\n",
        "print(f'RMSE (Clean): {rmse_clean:.2f} VND')\n",
        "print(f'MAPE (Clean): {mape:.2f}%')\n",
        "print('='*50 + '\\n')\n"
    ]
}

insert_idx = len(nb['cells'])
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 0 and 'Push Outputs' in cell['source'][0]:
        insert_idx = i
        break

nb['cells'].insert(insert_idx, new_cell)

with open('train_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
