"""Проверки границ CV и обучения preprocessing внутри каждого фолда."""

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.search_models import search_candidates, select_candidate
from scripts.train_baseline import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, split_data
from scripts.tune_models import build_candidate, make_cv_splits, run_cv


def example_data(size=100):
    frame = pd.DataFrame({name: [f"category-{i}" for i in range(size)]
                          for name in CATEGORICAL_FEATURES})
    for name in NUMERIC_FEATURES:
        frame[name] = np.arange(size, dtype=float) ** 2
    frame.loc[::7, "TotalCharges"] = np.nan
    frame["customerID"] = [f"id-{i}" for i in range(size)]
    frame["Churn"] = [i % 2 for i in range(size)]
    return frame


class TuningTests(unittest.TestCase):
    def test_random_search_fits_only_cv_training_rows_and_reports_held_out_scores(self):
        frame = split_data(example_data(60))["train"]
        folds = make_cv_splits(frame["Churn"])
        fitted_indices = []
        original_fit = Pipeline.fit

        def checked_fit(model, features, target, **kwargs):
            result = original_fit(model, features, target, **kwargs)
            if "classifier" in model.named_steps:
                fitted_indices.append(tuple(features.index))
                preprocessing = model.named_steps["preprocessing"]
                numeric = preprocessing.named_transformers_["numeric"]
                expected = features[NUMERIC_FEATURES].median().to_numpy()
                np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, expected)
                filled = features[NUMERIC_FEATURES].fillna(dict(zip(NUMERIC_FEATURES, expected)))
                np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, filled.mean())
                encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
                learned = set(encoder.categories_[CATEGORICAL_FEATURES.index("Contract")])
                self.assertEqual(learned, set(features["Contract"]))
            return result

        with patch.object(Pipeline, "fit", checked_fit):
            candidates = search_candidates(
                "logistic_regression", frame[FEATURES], frame["Churn"], folds,
                distributions={"classifier__C": [0.1, 10.0]}, n_iter=2, n_jobs=1, verbose=0,
            )
        # Ровно 2 × 10 fit, без дополнительного refit и без внешних validation/test.
        expected_indices = [tuple(frame.iloc[fit].index) for fit, _ in folds] * 2
        self.assertEqual(Counter(fitted_indices), Counter(expected_indices))
        for candidate in candidates:
            manual = run_cv(build_candidate("logistic_regression", candidate["parameters"]),
                            frame[FEATURES], frame["Churn"], folds)
            for metric in ("average_precision", "roc_auc"):
                np.testing.assert_allclose(candidate[metric]["fold_scores"], manual[f"test_{metric}"])

    def test_search_keeps_previous_winner_unless_new_cv_score_is_better(self):
        def candidate(score):
            return {"average_precision": {"mean": score}}

        self.assertEqual(select_candidate([candidate(0.7), candidate(0.69)]), 0)
        self.assertEqual(select_candidate([candidate(0.7), candidate(0.7)]), 0)
        self.assertEqual(select_candidate([candidate(0.7), candidate(0.71)]), 1)
        for invalid in ([], [candidate(float("nan"))], [candidate(float("inf"))]):
            with self.assertRaises(ValueError):
                select_candidate(invalid)

    def test_cv_folds_cover_train_only_and_are_repeatable(self):
        parts = split_data(example_data())
        train = parts["train"]
        folds = make_cv_splits(train["Churn"])
        repeated = make_cv_splits(train["Churn"])
        outer_ids = set(parts["validation"]["customerID"]) | set(parts["test"]["customerID"])
        for repeat in range(2):
            held_out_all = []
            for fit, held_out in folds[repeat * 5:(repeat + 1) * 5]:
                self.assertFalse(set(fit) & set(held_out))
                self.assertEqual(set(fit) | set(held_out), set(range(len(train))))
                self.assertFalse(set(train.iloc[fit]["customerID"]) & outer_ids)
                self.assertFalse(set(train.iloc[held_out]["customerID"]) & outer_ids)
                self.assertEqual(train.iloc[held_out]["Churn"].mean(), 0.5)
                held_out_all.extend(held_out)
            self.assertEqual(sorted(held_out_all), list(range(len(train))))
        for original, other in zip(folds, repeated, strict=True):
            for first, second in zip(original, other, strict=True):
                np.testing.assert_array_equal(first, second)

    def test_cv_estimators_learn_only_their_own_fit_fold(self):
        frame = example_data(40)
        folds = make_cv_splits(frame["Churn"], seeds=(42,), n_splits=2)
        scores = run_cv(build_candidate("logistic_regression", {"C": 1}),
                        frame[FEATURES], frame["Churn"], folds, return_estimator=True)
        self.assertEqual(len(scores["estimator"]), 2)
        for model, (fit, held_out) in zip(scores["estimator"], folds, strict=True):
            preprocessing = model.named_steps["preprocessing"]
            numeric = preprocessing.named_transformers_["numeric"]
            expected = frame.iloc[fit][NUMERIC_FEATURES].median().to_numpy()
            np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, expected)
            filled = frame.iloc[fit][NUMERIC_FEATURES].fillna(dict(zip(NUMERIC_FEATURES, expected)))
            np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, filled.mean())
            encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
            contract_index = CATEGORICAL_FEATURES.index("Contract")
            learned = set(encoder.categories_[contract_index])
            self.assertEqual(learned, set(frame.iloc[fit]["Contract"]))
            self.assertFalse(learned & set(frame.iloc[held_out]["Contract"]))
            self.assertTrue(np.isfinite(model.predict_proba(frame.iloc[held_out][FEATURES])).all())


if __name__ == "__main__":
    unittest.main()
