"""Leakage-safe CEEMDAN decomposition expert for electricity prices."""
import numpy as np
import lightgbm as lgb


class CEEMDANForecaster:
    """Decompose fold-local targets and model each component independently."""

    def __init__(self, n_imfs_max=6, trials=50):
        self.n_imfs_max = n_imfs_max
        self.trials = trials
        self.imf_models = []
        self.n_imfs_actual = 0

    def _get_imf_model(self):
        return lgb.LGBMRegressor(
            objective="mae",
            learning_rate=0.03,
            num_leaves=63,
            n_estimators=500,
            colsample_bytree=0.8,
            min_child_samples=30,
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=42,
            verbosity=-1,
        )

    def _decompose(self, y_series):
        try:
            from PyEMD import CEEMDAN

        except ImportError as exc:
            raise ImportError(
                "CEEMDAN requires EMD-signal from requirements.txt"
            ) from exc

        ceemdan = CEEMDAN(trials=self.trials, epsilon=0.005, parallel=False)
        ceemdan.noise_seed(42)
        imfs = ceemdan(np.asarray(y_series, dtype=float))

        if len(imfs) > self.n_imfs_max:
            merged_residual = np.sum(imfs[self.n_imfs_max - 1 :], axis=0)
            imfs = np.vstack(
                [imfs[: self.n_imfs_max - 1], merged_residual.reshape(1, -1)]
            )

        return imfs

    def fit(self, X_train, y_train):
        y_values = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
        imfs = self._decompose(y_values)
        self.n_imfs_actual = len(imfs)
        self.imf_models = []
        for imf_target in imfs:
            model = self._get_imf_model()
            model.fit(X_train, imf_target)
            self.imf_models.append(model)
        return self

    def predict(self, X_test):
        total_pred = np.zeros(len(X_test), dtype=float)
        for model in self.imf_models:
            total_pred += model.predict(X_test)
        return total_pred


# Preserve loading of model artifacts created under the old class name.
ICEEMDANForecaster = CEEMDANForecaster
