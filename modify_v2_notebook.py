import json

with open('v2/train_single_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Change num_leaves in Cell 2
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'LGB_PARAMS = {' in source:
            source = source.replace("'num_leaves': 127", "'num_leaves': 255")
            nb['cells'][i]['source'] = [line + '\n' for line in source.split('\n')]
            break

# Insert Flat array generation after cell 12
flat_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- VERTICAL FLATTENING FOR GLOBAL MODEL ---\n",
        "N_days, F = X_daily.shape\n",
        "_, C = Y_daily.shape\n",
        "\n",
        "X_flat = []\n",
        "Y_res_flat = []\n",
        "Y_base_flat = []\n",
        "Y_daily_flat = []\n",
        "dates_flat = []\n",
        "\n",
        "for i in range(N_days):\n",
        "    for c in range(C):\n",
        "        row = X_daily[i].copy()\n",
        "        sin_h = np.sin(2 * np.pi * c / 48.0)\n",
        "        cos_h = np.cos(2 * np.pi * c / 48.0)\n",
        "        row = np.append(row, [c, sin_h, cos_h])\n",
        "        X_flat.append(row)\n",
        "        \n",
        "        Y_res_flat.append(Y_res[i, c])\n",
        "        Y_base_flat.append(Y_base[i, c])\n",
        "        Y_daily_flat.append(Y_daily[i, c])\n",
        "        dates_flat.append(dates_arr[i])\n",
        "\n",
        "X_daily = np.array(X_flat, dtype=np.float32)\n",
        "Y_res = np.array(Y_res_flat, dtype=np.float32)\n",
        "Y_base = np.array(Y_base_flat, dtype=np.float32)\n",
        "Y_daily = np.array(Y_daily_flat, dtype=np.float32)\n",
        "dates_arr = np.array(dates_flat)\n",
        "\n",
        "feature_cols = feature_cols + ['target_cycle_id', 'target_sin_hour', 'target_cos_hour']\n",
        "print(f'Flattened X: {X_daily.shape}, Y_res: {Y_res.shape}')\n"
    ]
}

# Find cell 12 which contains # --- RESIDUAL LEARNING SETUP ---
insert_idx = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'Residuals prepared' in ''.join(cell['source']):
        insert_idx = i + 1
        break

nb['cells'].insert(insert_idx, flat_cell)

# Now we must find and replace Cell 15 (Training), Cell 17 (Eval), Cell 20 (Fimp), Cell 23 (CV), Cell 25 (Save)
def replace_cell_source(keyword, new_source):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and keyword in ''.join(cell['source']):
            nb['cells'][i]['source'] = [line + '\n' for line in new_source.split('\n')]
            return

replace_cell_source('models = {}', """%%time
# SINGLE GLOBAL MODEL
train_ds = lgb.Dataset(X_tr, label=Y_tr, feature_name=feature_cols, free_raw_data=True)
val_ds   = lgb.Dataset(X_vl, label=Y_vl, reference=train_ds, free_raw_data=True)

global_model = lgb.train(
    LGB_PARAMS, train_ds, num_boost_round=NUM_BOOST_ROUND,
    valid_sets=[val_ds],
    callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(10)],
)

val_mae = global_model.best_score['valid_0']['l1']
print(f'\\nDone. Global model trained. Val MAE: {val_mae:.4f}')""")

replace_cell_source('Y_pred_res = np.zeros_like(Y_test)', """Y_pred_res_flat = global_model.predict(X_test)
Y_pred_log_flat = np.log1p(Y_base_test) + Y_pred_res_flat
Y_pred_flat = np.clip(np.expm1(Y_pred_log_flat), 0, PRICE_CAP)
Y_test_actual_flat = Y_test + Y_base_test

# Reshape back to compute cycle-specific metrics properly
N_test_days = len(Y_pred_flat) // CYCLES_PER_DAY
Y_pred = Y_pred_flat.reshape(N_test_days, CYCLES_PER_DAY)
Y_test_actual = Y_test_actual_flat.reshape(N_test_days, CYCLES_PER_DAY)

test_rmse = np.sqrt(mean_squared_error(Y_test_actual_flat, Y_pred_flat))
test_mae  = mean_absolute_error(Y_test_actual_flat, Y_pred_flat)

print(f'Test RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}')
print(f'Naive RMSE: {naive_rmse:.2f} | MAE: {naive_mae:.2f}')
print(f'Improvement: {(1 - test_rmse/naive_rmse)*100:.1f}% RMSE, {(1 - test_mae/naive_mae)*100:.1f}% MAE')

cycle_rmses, cycle_maes = [], []
for k in range(CYCLES_PER_DAY):
    cycle_rmses.append(np.sqrt(mean_squared_error(Y_test_actual[:, k], Y_pred[:, k])))
    cycle_maes.append(mean_absolute_error(Y_test_actual[:, k], Y_pred[:, k]))""")

replace_cell_source('fimp = pd.DataFrame', """residuals = Y_test_actual_flat - Y_pred_flat
fimp = pd.DataFrame({'feature': global_model.feature_name(), 'importance': global_model.feature_importance(importance_type='gain')})
fimp = fimp.sort_values('importance', ascending=False)
print(f'Residuals: mean={residuals.mean():.2f}, std={residuals.std():.2f}')
print(fimp.head(20).to_string(index=False))""")

replace_cell_source('cv_splits = [', """cv_splits = [
    {'name': 'Fold 1', 'train_end': '2024-06-30', 'val_start': '2024-07-01', 'val_end': '2025-03-31'},
    {'name': 'Fold 2', 'train_end': '2025-06-30', 'val_start': '2025-07-01', 'val_end': '2026-03-31'},
    {'name': 'Final',  'train_end': '2026-03-31', 'val_start': '2026-04-01', 'val_end': '2026-06-19'},
]

cv_results = []
for split in cv_splits:
    tr_mask = dates_pd <= split['train_end']
    vl_mask = (dates_pd >= split['val_start']) & (dates_pd <= split['val_end'])
    if vl_mask.sum() == 0: continue
    Xtr, Ytr = X_daily[tr_mask], Y_res[tr_mask]
    Xvl, Yvl = X_daily[vl_mask], Y_res[vl_mask]
    Ybase_vl = Y_base[vl_mask]
    nv = max(20 * 48, int(len(Xtr) * 0.1))
    print(f'{split["name"]}: train={tr_mask.sum()//48}, val={vl_mask.sum()//48} days')
    
    td = lgb.Dataset(Xtr[:-nv], label=Ytr[:-nv], feature_name=feature_cols, free_raw_data=True)
    vd = lgb.Dataset(Xtr[-nv:], label=Ytr[-nv:], reference=td, free_raw_data=True)
    m = lgb.train(LGB_PARAMS, td, num_boost_round=NUM_BOOST_ROUND,
                  valid_sets=[vd], callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)])
    Ypred_cv_res_flat = m.predict(Xvl)
    del td, vd, m; gc.collect()
    
    Ypred_cv_log = np.log1p(Ybase_vl) + Ypred_cv_res_flat
    Ypred_cv_flat = np.clip(np.expm1(Ypred_cv_log), 0, PRICE_CAP)
    Yvl_actual_flat = Yvl + Ybase_vl
    
    rmse = np.sqrt(mean_squared_error(Yvl_actual_flat, Ypred_cv_flat))
    mae  = mean_absolute_error(Yvl_actual_flat, Ypred_cv_flat)
    cv_results.append({'fold': split['name'], 'rmse': rmse, 'mae': mae})
    print(f'  RMSE: {rmse:.2f} | MAE: {mae:.2f}\\n')""")

replace_cell_source('model_dir = OUTPUT_DIR', """model_dir = OUTPUT_DIR / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

global_model.save_model(str(model_dir / 'lgb_global.txt'))

metadata = {
    'feature_names': feature_cols,
    'params': LGB_PARAMS,
    'num_boost_round': NUM_BOOST_ROUND,
    'test_rmse': float(test_rmse), 'test_mae': float(test_mae),
    'naive_rmse': float(naive_rmse), 'naive_mae': float(naive_mae),
    'cv_results': cv_results,
    'cycle_rmses': [float(r) for r in cycle_rmses],
    'cycle_maes': [float(m) for m in cycle_maes],
}
with open(model_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Saved global model to {model_dir}')
for f in sorted(OUTPUT_DIR.rglob('*')):
    if f.is_file(): print(f'  {f.relative_to(OUTPUT_DIR)} ({f.stat().st_size / 1024:.0f} KB)')""")


# Need to clean up multiple newlines created by split
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i in range(len(cell['source'])):
            cell['source'][i] = cell['source'][i].replace('\n\n', '\n')

with open('v2/train_single_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
