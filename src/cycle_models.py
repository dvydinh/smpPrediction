import lightgbm as lgb
import numpy as np


def _cycles(index):
    return index.hour * 2 + (index.minute == 30).astype(int)


class CleanCycleResidualForecaster:
    """Cycle-specific residual expert aligned with the clean evaluation regime."""

    def __init__(self, n_estimators=1200):
        self.n_estimators = n_estimators
        self.models = {}
        self.best_iterations_ = {}

    @staticmethod
    def _baseline(X):
        if "smp_same_cycle_1d" not in X.columns:
            raise ValueError("smp_same_cycle_1d is required")
        return X["smp_same_cycle_1d"].to_numpy(dtype=float)

    def _new_model(self):
        return lgb.LGBMRegressor(
            objective="mae",
            learning_rate=0.02,
            num_leaves=15,
            max_depth=5,
            n_estimators=self.n_estimators,
            min_child_samples=24,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
        )

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_val=None,
        y_val=None,
        val_weight=None,
    ):
        cycle_values = np.asarray(_cycles(X.index), dtype=int)
        baseline = self._baseline(X)
        target = y.to_numpy(dtype=float)
        clean = (target > 500.0) & (target < 2500.0)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        if X_val is not None:
            val_cycles = np.asarray(_cycles(X_val.index), dtype=int)
            val_target = y_val.to_numpy(dtype=float)
            val_clean = (val_target > 500.0) & (val_target < 2500.0)
            val_baseline = self._baseline(X_val)
            validation_weights = None if val_weight is None else np.asarray(val_weight, dtype=float)

        self.models = {}
        self.best_iterations_ = {}
        for cycle in range(48):
            train_mask = clean & (cycle_values == cycle)
            if train_mask.sum() < 100:
                raise ValueError(f"Not enough clean samples for cycle {cycle}")
            model = self._new_model()
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["sample_weight"] = weights[train_mask]

            if X_val is not None:
                validation_mask = val_clean & (val_cycles == cycle)
                if validation_mask.sum() >= 20:
                    fit_kwargs["eval_set"] = [
                        (
                            X_val.iloc[np.flatnonzero(validation_mask)],
                            val_target[validation_mask] - val_baseline[validation_mask],
                        )
                    ]
                    if validation_weights is not None:
                        fit_kwargs["eval_sample_weight"] = [validation_weights[validation_mask]]
                    fit_kwargs["callbacks"] = [lgb.early_stopping(60, verbose=False)]

            model.fit(
                X.iloc[np.flatnonzero(train_mask)],
                target[train_mask] - baseline[train_mask],
                **fit_kwargs,
            )
            self.models[cycle] = model
            self.best_iterations_[cycle] = int(model.best_iteration_ or self.n_estimators)
        return self

    def predict(self, X):
        cycle_values = np.asarray(_cycles(X.index), dtype=int)
        result = self._baseline(X).copy()
        for cycle, model in self.models.items():
            mask = cycle_values == cycle
            if mask.any():
                result[mask] += model.predict(X.iloc[np.flatnonzero(mask)])
        return result
