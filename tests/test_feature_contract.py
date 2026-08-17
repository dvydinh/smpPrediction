import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import add_engineered_features
from src.feature_policy import OBSERVED_ONLY_COLUMNS, select_production_features


class FeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range("2025-01-01", periods=48 * 40, freq="30min")
        values = np.arange(len(index), dtype=float)
        raw = pd.DataFrame(
            {
                "smp_system_price": values,
                "smp_north_price": values + 10.0,
                "smp_central_price": values + 20.0,
                "smp_south_price": values + 30.0,
                "load_total_mw": values + 1000.0,
                "load_north_mw": values + 2000.0,
                "load_central_mw": values + 3000.0,
                "load_south_mw": values + 4000.0,
            },
            index=index,
        )
        cls.raw = raw
        cls.features = add_engineered_features(raw)

    def test_blindspot_lag_switches_after_cycle_15(self):
        day = self.features.index.normalize().unique()[35]
        morning = day
        afternoon = day + pd.Timedelta(hours=8)
        self.assertEqual(
            self.features.loc[morning, "smp_same_cycle_1d"],
            self.raw.loc[morning - pd.Timedelta(days=1), "smp_system_price"],
        )
        self.assertEqual(
            self.features.loc[afternoon, "smp_same_cycle_1d"],
            self.raw.loc[afternoon - pd.Timedelta(days=2), "smp_system_price"],
        )

    def test_snapshot_statistics_are_constant_within_target_day(self):
        day = self.features.index.normalize().unique()[35]
        values = self.features.loc[day:day + pd.Timedelta(hours=23, minutes=30), "smp_rolling_std_1d"]
        self.assertEqual(values.nunique(), 1)

    def test_observed_target_day_columns_are_blocked(self):
        selected = select_production_features(self.features)
        self.assertFalse(set(selected).intersection(OBSERVED_ONLY_COLUMNS))


if __name__ == "__main__":
    unittest.main()
