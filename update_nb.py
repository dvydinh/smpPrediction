import json

with open('train_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "'learning_rate': 0.01" in line:
                cell['source'][i] = line.replace("'learning_rate': 0.01", "'learning_rate': 0.005")
            elif "EARLY_STOPPING = 50" in line:
                cell['source'][i] = line.replace("EARLY_STOPPING = 50", "EARLY_STOPPING = 200")
            elif "Y_res = Y_daily - Y_base" in line:
                cell['source'][i] = line.replace("Y_res = Y_daily - Y_base", "Y_res = np.log1p(Y_daily) - np.log1p(Y_base)")
            elif "Y_pred = np.clip(Y_pred_res + Y_base_test, 0, PRICE_CAP)" in line:
                cell['source'][i] = line.replace("Y_pred = np.clip(Y_pred_res + Y_base_test, 0, PRICE_CAP)", "Y_pred_log = np.log1p(Y_base_test) + Y_pred_res\n    Y_pred = np.clip(np.expm1(Y_pred_log), 0, PRICE_CAP)")
            elif "Ypred_cv = np.clip(Ypred_cv_res + Ybase_vl, 0, PRICE_CAP)" in line:
                cell['source'][i] = line.replace("Ypred_cv = np.clip(Ypred_cv_res + Ybase_vl, 0, PRICE_CAP)", "Ypred_cv_log = np.log1p(Ybase_vl) + Ypred_cv_res\n        Ypred_cv = np.clip(np.expm1(Ypred_cv_log), 0, PRICE_CAP)")

with open('train_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
