from __future__ import annotations
import gc
import json
import logging
from pathlib import Path
import numpy as np
import lightgbm as lgb
from vcgm import config as cfg

logger = logging.getLogger(__name__)


class DirectForecaster:

    def __init__(self, params=None, num_boost_round=cfg.LGB_NUM_BOOST_ROUND, early_stopping=cfg.LGB_EARLY_STOPPING):
        self.params = dict(params or cfg.LGB_PARAMS)
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.models: dict[int, lgb.Booster] = {}
        self.feature_names: list[str] = []

    def fit(self, X_train, Y_train, feature_names, val_fraction=0.1, save_dir=cfg.MODEL_DIR):
        self.feature_names = feature_names
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        n = X_train.shape[0]
        n_val = max(20, int(n * val_fraction))
        X_tr, Y_tr = X_train[:-n_val], Y_train[:-n_val]
        X_vl, Y_vl = X_train[-n_val:], Y_train[-n_val:]

        diagnostics = {}

        for k in range(cfg.CYCLES_PER_DAY):
            train_ds = lgb.Dataset(X_tr, label=Y_tr[:, k], feature_name=feature_names, free_raw_data=True)
            val_ds = lgb.Dataset(X_vl, label=Y_vl[:, k], reference=train_ds, free_raw_data=True)

            model = lgb.train(
                self.params, train_ds, num_boost_round=self.num_boost_round,
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(self.early_stopping), lgb.log_evaluation(0)],
            )

            diag = {"n_trees": model.num_trees(), "val_mae": float(model.best_score["valid_0"]["l1"])}
            diagnostics[k] = diag
            model.save_model(str(save_dir / f"lgb_cycle_{k:02d}.txt"))

            if k % 8 == 0 or k == 47:
                t = f"{k // 2:02d}:{(k % 2) * 30:02d}"
                logger.info("  M_%02d (%s)  trees=%d  val_MAE=%.1f", k, t, diag["n_trees"], diag["val_mae"])

            model.free_dataset()
            del model, train_ds, val_ds
            gc.collect()

        meta = {"feature_names": self.feature_names, "params": self.params, "num_boost_round": self.num_boost_round}
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("48 models saved to %s", save_dir)
        self._load_from_dir(save_dir)
        return diagnostics

    def predict(self, X):
        if not self.models:
            raise RuntimeError("No models loaded")
        out = np.zeros((X.shape[0], cfg.CYCLES_PER_DAY))
        for k in range(cfg.CYCLES_PER_DAY):
            out[:, k] = self.models[k].predict(X)
        return np.clip(out, 0, cfg.PRICE_CAP_VND)

    def save(self, directory=cfg.MODEL_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for k, model in self.models.items():
            model.save_model(str(directory / f"lgb_cycle_{k:02d}.txt"))
        meta = {"feature_names": self.feature_names, "params": self.params, "num_boost_round": self.num_boost_round}
        with open(directory / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _load_from_dir(self, directory):
        self.models.clear()
        for k in range(cfg.CYCLES_PER_DAY):
            self.models[k] = lgb.Booster(model_file=str(Path(directory) / f"lgb_cycle_{k:02d}.txt"))

    @classmethod
    def load(cls, directory=cfg.MODEL_DIR):
        directory = Path(directory)
        with open(directory / "metadata.json") as f:
            meta = json.load(f)
        obj = cls(params=meta["params"], num_boost_round=meta["num_boost_round"])
        obj.feature_names = meta["feature_names"]
        obj._load_from_dir(directory)
        logger.info("Loaded %d models from %s", len(obj.models), directory)
        return obj
