import lightgbm as lgb
import numpy as np


class RegimeForecaster:
    """Two-part forecaster for collapse and normal price regimes."""

    def __init__(self, low_threshold=500.0, n_estimators=800, fit_normal=True):
        self.low_threshold = low_threshold
        self.fit_normal = bool(fit_normal)
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
        self.gate_threshold_ = 1.1
        self.gate_ramp_ = 0.0
        self.gate_validation_scores_ = {}

    def _new_classifier(self, scale_pos_weight=1.0):
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
            scale_pos_weight=float(scale_pos_weight),
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

        positive = int(low.sum())
        negative = int((~low).sum())
        scale_pos_weight = np.clip(negative / positive, 1.0, 20.0)
        self.classifier = self._new_classifier(scale_pos_weight)
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

        if self.fit_normal:
            self.normal_model = self._fit_regressor(
                "normal", X, target, ~low, sample_weight, X_val, y_val, val_weight
            )
        self.low_model = self._fit_regressor(
            "low", X, target, low, sample_weight, X_val, y_val, val_weight
        )
        return self

    def fit_calibrated(
        self,
        X,
        y,
        sample_weight=None,
        calibration_days=365,
    ):
        """Tune the collapse decision on a trailing historical block, then refit."""
        index = X.index
        cutoff = index.max().normalize() - np.timedelta64(int(calibration_days), "D")
        calibration = index >= cutoff
        development = ~calibration
        development_target = y.iloc[np.flatnonzero(development)].to_numpy(dtype=float)
        calibration_target = y.iloc[np.flatnonzero(calibration)].to_numpy(dtype=float)

        enough_history = (
            development.sum() >= 512
            and calibration.sum() >= 96
            and (development_target <= self.low_threshold).sum() >= 100
            and (calibration_target <= self.low_threshold).sum() >= 20
        )
        if enough_history:
            probe = RegimeForecaster(
                low_threshold=self.low_threshold,
                n_estimators=self.n_estimators,
                fit_normal=self.fit_normal,
            )
            probe_weight = None
            if sample_weight is not None:
                probe_weight = np.asarray(sample_weight, dtype=float)[development]
                probe_weight = probe_weight / max(probe_weight.max(), 1e-12)
            probe.fit(
                X.iloc[np.flatnonzero(development)],
                y.iloc[np.flatnonzero(development)],
                sample_weight=probe_weight,
            )
            probability, _, _ = probe.predict_components(
                X.iloc[np.flatnonzero(calibration)]
            )
            self._select_gate(calibration_target, probability)

        threshold = self.gate_threshold_
        ramp = self.gate_ramp_
        scores = dict(self.gate_validation_scores_)
        self.fit(X, y, sample_weight=sample_weight)
        self.gate_threshold_ = threshold
        self.gate_ramp_ = ramp
        self.gate_validation_scores_ = scores
        return self

    def _select_gate(self, actual, probability):
        actual = np.asarray(actual, dtype=float)
        probability = np.asarray(probability, dtype=float)
        event = actual <= self.low_threshold
        prevalence = float(event.mean())
        minimum_precision = max(0.25, min(0.60, 1.5 * prevalence))
        best = None
        self.gate_validation_scores_ = {}

        for threshold in np.arange(0.05, 0.991, 0.025):
            predicted = probability >= threshold
            true_positive = int(np.sum(event & predicted))
            false_positive = int(np.sum(~event & predicted))
            false_negative = int(np.sum(event & ~predicted))
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            fpr = false_positive / max((~event).sum(), 1)
            beta_squared = 4.0
            f2 = (
                (1.0 + beta_squared) * precision * recall
                / max(beta_squared * precision + recall, 1e-12)
            )
            self.gate_validation_scores_[f"t{threshold:.3f}"] = {
                "precision": float(precision),
                "recall": float(recall),
                "fpr": float(fpr),
                "f2": float(f2),
            }
            if precision < minimum_precision or fpr > 0.0025:
                continue
            candidate = (f2, recall, precision, -threshold)
            if best is None or candidate > best[0]:
                best = (candidate, float(threshold))

        if best is not None:
            self.gate_threshold_ = best[1]

    def gate_weight(self, probability):
        probability = np.asarray(probability, dtype=float)
        if self.gate_threshold_ > 1.0:
            return np.zeros(len(probability), dtype=float)
        if self.gate_ramp_ <= 0.0:
            return (probability >= self.gate_threshold_).astype(float)
        lower = self.gate_threshold_ - self.gate_ramp_
        upper = self.gate_threshold_ + self.gate_ramp_
        return np.clip((probability - lower) / (upper - lower), 0.0, 1.0)

    def apply_gate(self, normal_prediction, probability, low_prediction):
        weight = self.gate_weight(probability)
        low_prediction = np.clip(
            np.asarray(low_prediction, dtype=float),
            0.0,
            self.low_threshold,
        )
        normal_prediction = np.asarray(normal_prediction, dtype=float)
        return (1.0 - weight) * normal_prediction + weight * low_prediction

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
        normal_prediction = (
            self.normal_model.predict(X)
            if self.normal_model is not None
            else np.full(len(X), np.nan, dtype=float)
        )
        low_prediction = self.low_model.predict(X)
        return low_probability, normal_prediction, low_prediction

    def predict(self, X):
        probability, normal, low = self.predict_components(X)
        return (1.0 - probability) * normal + probability * low
