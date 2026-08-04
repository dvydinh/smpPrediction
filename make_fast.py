import time
print("Starting script...")
with open('eval_mape.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace("print(f'X: {X_daily.shape}'", "print('Data built'); print(f'X: {X_daily.shape}'")
code = code.replace("models[k] = lgb.Booster(", "print(f'Loading model {k}...'); models[k] = lgb.Booster(")
with open('eval_mape_fast.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("eval_mape_fast.py created.")
