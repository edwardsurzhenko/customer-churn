"""Проверки границ экспериментального заполнения и учёта ошибок по группам."""

import unittest

import numpy as np
import pandas as pd

from scripts.analyze_errors import segment_counts, zero_new_customer_charges


class ErrorAnalysisTests(unittest.TestCase):
    def test_zero_rule_only_changes_missing_charge_with_known_zero_tenure(self):
        frame = pd.DataFrame({
            "tenure": [0.0, 0.0, 3.0, np.nan],
            "TotalCharges": [np.nan, 25.0, np.nan, np.nan],
            "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
        }, index=[7, 2, 13, 5])
        original = frame.copy(deep=True)
        changed = zero_new_customer_charges(frame)
        pd.testing.assert_frame_equal(frame, original)
        expected = original.copy()
        expected.loc[7, "TotalCharges"] = 0.0
        pd.testing.assert_frame_equal(changed, expected)

    def test_segment_errors_account_for_every_client_and_undefined_rates(self):
        features = pd.DataFrame({
            "Contract": ["monthly", "monthly", "annual", "annual"],
            "tenure": [0.0, 12.0, 24.0, np.nan],
        }, index=[10, 20, 40, 70])
        target = pd.Series([1, 0, 1, 0], index=features.index)
        results = segment_counts(features, target, [0.5, 0.8, 0.2, 0.1])
        for groups in results.values():
            self.assertEqual(sum(group["count"] for group in groups.values()), 4)
            self.assertEqual(sum(group["positives"] for group in groups.values()), 2)
            for error in ["tp", "fp", "fn", "tn"]:
                self.assertEqual(sum(group[error] for group in groups.values()), 1)
        self.assertEqual(results["Contract"]["monthly"]["precision"], 0.5)
        self.assertIsNone(results["Contract"]["annual"]["precision"])
        self.assertIsNone(results["tenure_group"]["missing"]["recall"])
        self.assertEqual(results["tenure_group"]["0–12"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
