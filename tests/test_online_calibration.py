import unittest

import numpy as np
import pandas as pd

from src.online_calibration import (
    apply_online_adjustment,
    apply_single_day_adjustment,
    target_metrics,
)


class OnlineCalibrationTests(unittest.TestCase):
    def test_target_metrics_report_collapse_performance(self):
        metrics = target_metrics(
            np.array([0.0, 1000.0, 2000.0]),
            np.array([100.0, 900.0, 2000.0]),
        )
        self.assertEqual(metrics["collapse_samples"], 1)
        self.assertEqual(metrics["collapse_precision"], 1.0)
        self.assertEqual(metrics["collapse_recall"], 1.0)
        self.assertEqual(metrics["collapse_mae"], 100.0)

    def test_future_actual_does_not_change_earlier_forecasts(self):
        index = pd.date_range("2025-01-01", periods=48 * 10, freq="30min")
        base = np.full(len(index), 1000.0)
        actual = np.full(len(index), 1100.0)
        changed = actual.copy()
        changed[48 * 7:] = 1700.0

        first = apply_online_adjustment(
            base,
            index,
            actual,
            bias_days=7,
            cap_days=0,
            cap_ratio=None,
        )
        second = apply_online_adjustment(
            base,
            index,
            changed,
            bias_days=7,
            cap_days=0,
            cap_ratio=None,
        )
        np.testing.assert_allclose(first[:48 * 8], second[:48 * 8])

    def test_cap_snap_uses_only_completed_history(self):
        history_index = pd.date_range(
            "2025-01-01",
            periods=48 * 28,
            freq="30min",
        )
        history_actual = pd.Series(1700.0, index=history_index)
        history_base = pd.Series(1650.0, index=history_index)
        target_index = pd.date_range(
            "2025-01-29",
            periods=48,
            freq="30min",
        )
        prediction = apply_single_day_adjustment(
            np.full(48, 1650.0),
            target_index,
            history_actual,
            history_base,
            bias_days=0,
            cap_days=28,
            cap_ratio=0.95,
        )
        np.testing.assert_allclose(prediction, 1700.0)

    def test_single_day_requires_48_cycles(self):
        with self.assertRaises(ValueError):
            apply_single_day_adjustment(
                np.full(47, 1000.0),
                pd.date_range("2025-01-01", periods=47, freq="30min"),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
