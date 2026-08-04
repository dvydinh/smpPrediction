import logging
import pandas as pd
from vcgm import config as cfg
from vcgm.features import pipeline
from vcgm.models.direct_forecast import DirectForecaster
from vcgm.models import evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    df = pd.read_parquet(cfg.DESIGN_MATRIX_FILE)
    feature_cols = [c for c in df.columns if c not in ["smp_system_price", "cycle_id"]]
    X, Y, dates = pipeline.build_daily_matrices(df, feature_cols)

    results = evaluation.walk_forward_cv(DirectForecaster, X, Y, dates, feature_cols, cfg.CV_SPLITS)
    for r in results:
        print(f"{r['fold']}: RMSE={r['overall_rmse']:.2f} MAE={r['overall_mae']:.2f}")
    evaluation.save_metrics({"cv_results": results})

if __name__ == "__main__":
    main()
