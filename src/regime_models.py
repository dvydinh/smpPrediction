import lightgbm as lgb
import numpy as np


class RegimeForecaster:
    """Two-part forecaster for collapse and normal price regimes."""

    def __init__(self, low_threshold=500.0, n_estimators=800):
        self.low_threshold = low_threshold
        if isinstance(n_estimators, dict):
            self.n_estimators = dict(n_estimators)
        else:
            self.n_estimators = {
                "classifier": int(n_estimators),
                "normal": int(n_estimators),
                "low": int(n_estimators),
            }
        self.classifier = None
        self.normal_model = None
        self.low_model = None
        self.best_iterations_ = {}

    def _new_classifier(self):
        return lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.02,
            num_leaves=31,
            max_depth=7,
            n_estimators=self.n_estimators["classifier"],
            min_child_samples=40,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
        )

    def _new_regressor(self, regime):
        return lgb.LGBMRegressor(
            objective="mae",
            learning_rate=0.02,
            num_leaves=31,
            max_depth=7,
            n_estimators=self.n_estimators[regime],
            min_child_samples=30,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=42,
            verbosity=-1,
            force_col_wise=True,
        )

    @staticmethod
    def _weights(sample_weight, mask=None):
        if sample_weight is None:
            return None
        values = np.asarray(sample_weight, dtype=float)
        return values if mask is None else values[mask]

    @staticmethod
    def _best_iteration(model, fallback):
        return int(getattr(model, "best_iteration_", 0) or fallback)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_val=None,
        y_val=None,
        val_weight=None,
    ):
        target = y.to_numpy(dtype=float)
        low = target <= self.low_threshold
        if low.sum() < 100 or (~low).sum() < 100:
            raise ValueError("Both price regimes require at least 100 samples")

        self.classifier = self._new_classifier()
        classifier_kwargs = {}
        if sample_weight is not None:
            classifier_kwargs["sample_weight"] = self._weights(sample_weight)
        if X_val is not None:
            validation_target = y_val.to_numpy(dtype=float)
            classifier_kwargs.update({
                "eval_set": [(X_val, validation_target <= self.low_threshold)],
                "eval_metric": "binary_logloss",
                "callbacks": [lgb.early_stopping(60, verbose=False)],
            })
            if val_weight is not None:
                classifier_kwargs["eval_sample_weight"] = [self._weights(val_weight)]
        self.classifier.fit(X, low.astype(int), **classifier_kwargs)
        self.best_iterations_["classifier"] = self._best_iteration(
            self.classifier,
            self.n_estimators["classifier"],
        )

        self.normal_model = self._fit_regressor(
            "normal", X, target, ~low, sample_weight, X_val, y_val, val_weight
        )
        self.low_model = self._fit_regressor(
            "low", X, target, low, sample_weight, X_val, y_val, val_weight
        )
        return self

    def _fit_regressor(
        self,
        regime,
        X,
        target,
        train_mask,
        sample_weight,
        X_val,
        y_val,
        val_weight,
    ):
        model = self._new_regressor(regime)
        kwargs = {}
        if sample_weight is not None:
            kwargs["sample_weight"] = self._weights(sample_weight, train_mask)
        if X_val is not None:
            validation_target = y_val.to_numpy(dtype=float)
            if regime == "normal":
                validation_mask = validation_target > self.low_threshold
            else:
                validation_mask = validation_target <= self.low_threshold
            if validation_mask.sum() >= 20:
                kwargs.update({
                    "eval_set": [(
                        X_val.iloc[np.flatnonzero(validation_mask)],
                        validation_target[validation_mask],
                    )],
                    "callbacks": [lgb.early_stopping(60, verbose=False)],
                })
                if val_weight is not None:
                    kwargs["eval_sample_weight"] = [
                        self._weights(val_weight, validation_mask)
                    ]
        model.fit(
            X.iloc[np.flatnonzero(train_mask)],
            target[train_mask],
            **kwargs,
        )
        self.best_iterations_[regime] = self._best_iteration(
            model,
            self.n_estimators[regime],
        )
        return model

    def predict_components(self, X):
        low_probability = self.classifier.predict_proba(X)[:, 1]
        normal_prediction = self.normal_model.predict(X)
        low_prediction = self.low_model.predict(X)
        return low_probability, normal_prediction, low_prediction

    def predict(self, X):
        probability, normal, low = self.predict_components(X)
        return (1.0 - probability) * normal + probability * low
