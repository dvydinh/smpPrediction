import logging
import pandas as pd
from vcgm import config as cfg
from vcgm.features import pipeline
from vcgm.models.direct_forecast import DirectForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    df = pd.read_parquet(cfg.DESIGN_MATRIX_FILE)
    feature_cols = [c for c in df.columns if c not in ["smp_system_price", "cycle_id"]]
    X, Y, dates = pipeline.build_daily_matrices(df, feature_cols)

    dates_pd = pd.to_datetime(dates)
    train_mask = dates_pd <= "2026-03-31"
    X_train, Y_train = X[train_mask], Y[train_mask]

    fc = DirectForecaster()
    fc.fit(X_train, Y_train, feature_cols)
    print("Done")

if __name__ == "__main__":
    main()
