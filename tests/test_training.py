"""Проверки разделения данных и отсутствия утечки через preprocessing."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.train_baseline import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
    build_logistic_model, load_data, split_data,
)


def sample_data(size=100):
    frame = pd.DataFrame({column: ["A"] * size for column in CATEGORICAL_FEATURES})
    for column in NUMERIC_FEATURES:
        frame[column] = np.arange(size, dtype=float)
    frame["customerID"] = [f"client-{index}" for index in range(size)]
    frame["Churn"] = [index % 2 for index in range(size)]
    return frame


class TrainingTests(unittest.TestCase):
    def test_splits_are_disjoint_complete_and_repeatable(self):
        frame = sample_data()
        parts = split_data(frame)
        repeated = split_data(frame)
        ids = {name: set(part["customerID"]) for name, part in parts.items()}
        self.assertFalse(ids["train"] & ids["validation"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["validation"] & ids["test"])
        self.assertEqual(set.union(*ids.values()), set(frame["customerID"]))
        self.assertEqual({name: len(part) for name, part in parts.items()},
                         {"train": 60, "validation": 20, "test": 20})
        for name in parts:
            self.assertEqual(parts[name]["customerID"].tolist(),
                             repeated[name]["customerID"].tolist())
            self.assertEqual(parts[name]["Churn"].mean(), 0.5)

    def test_validation_does_not_change_training_transformations(self):
        train = sample_data(6)
        train["TotalCharges"] = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
        model = build_logistic_model()
        model.fit(train[FEATURES], train["Churn"])
        preprocessing = model.named_steps["preprocessing"]
        numeric = preprocessing.named_transformers_["numeric"]
        statistics_before = numeric.named_steps["imputer"].statistics_.copy()
        means_before = numeric.named_steps["scaler"].mean_.copy()

        validation = sample_data(2)
        validation["TotalCharges"] = [1000000.0, np.nan]
        validation["Contract"] = ["unseen-contract", "unseen-contract"]
        probability = model.predict_proba(validation[FEATURES])

        self.assertTrue(np.isfinite(probability).all())
        self.assertEqual(statistics_before[NUMERIC_FEATURES.index("TotalCharges")], 4.0)
        np.testing.assert_array_equal(numeric.named_steps["imputer"].statistics_,
                                      statistics_before)
        np.testing.assert_array_equal(numeric.named_steps["scaler"].mean_, means_before)
        self.assertNotIn("customerID", preprocessing.feature_names_in_)
        self.assertNotIn("Churn", preprocessing.feature_names_in_)

    def test_duplicate_ids_are_rejected(self):
        frame = sample_data(6)
        frame["Churn"] = frame["Churn"].map({0: "No", 1: "Yes"})
        frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "уникальным"):
                load_data(path)


if __name__ == "__main__":
    unittest.main()
