"""Проверки ранжирования, метрик кампании и граничных случаев."""

import unittest

import numpy as np
import pandas as pd

from scripts.ranking import evaluate_top_k, random_selection_reference, rank_customers


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array(["d", "a", "e", "c", "b"])
        self.scores = np.array([0.6, 0.9, 0.2, 0.6, 0.6])
        self.target = np.array([1, 0, 0, 1, 0])

    def test_ranking_is_independent_of_input_order_and_has_no_target(self):
        result = rank_customers(self.ids, self.scores)
        self.assertEqual(result["customerID"].tolist(), ["a", "b", "c", "d", "e"])
        self.assertEqual(result["rank"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(result.columns.tolist(), ["rank", "customerID", "risk_probability"])
        order = [2, 4, 0, 3, 1]
        pd.testing.assert_frame_equal(result, rank_customers(self.ids[order], self.scores[order]))
        self.assertEqual(
            evaluate_top_k(self.ids, self.scores, self.target, [3]),
            evaluate_top_k(self.ids[order], self.scores[order], self.target[order], [3]),
        )

    def test_metrics_and_cutoff_tie_bounds_match_hand_worked_example(self):
        first, middle, full = evaluate_top_k(self.ids, self.scores, self.target, [1, 3, 5])
        self.assertEqual(first["true_positives"], 0)
        self.assertEqual(middle["true_positives"], 1)
        self.assertEqual(middle["false_positives"], 2)
        self.assertAlmostEqual(middle["precision_at_k"], 1 / 3)
        self.assertEqual(middle["recall_at_k"], 0.5)
        self.assertAlmostEqual(middle["lift_at_k"], 5 / 6)
        self.assertEqual(middle["cutoff_tie_size"], 3)
        self.assertEqual(middle["selected_from_cutoff_tie"], 2)
        self.assertEqual(middle["true_positive_bounds_within_cutoff_tie"], [1, 2])
        self.assertEqual(full["true_positives"], 2)
        self.assertEqual(full["precision_at_k"], 0.4)
        self.assertEqual(full["recall_at_k"], 1)
        self.assertEqual(full["lift_at_k"], 1)

    def test_random_reference_is_an_expectation_not_a_sample(self):
        first, full = random_selection_reference(5, 2, [1, 5])
        self.assertEqual(first["expected_true_positives"], 0.4)
        self.assertEqual(first["expected_precision_at_k"], 0.4)
        self.assertEqual(first["expected_recall_at_k"], 0.2)
        self.assertEqual(first["expected_lift_at_k"], 1)
        self.assertEqual(full["expected_true_positives"], 2)
        self.assertEqual(full["expected_recall_at_k"], 1)

    def test_all_ties_and_single_class_have_explicit_semantics(self):
        row = evaluate_top_k(self.ids, np.full(5, 0.5), self.target, [2])[0]
        self.assertEqual(row["true_positives"], 0)  # a, b выбраны по ID.
        self.assertEqual(row["true_positive_bounds_within_cutoff_tie"], [0, 2])
        for labels in [np.zeros(5), np.ones(5)]:
            row = evaluate_top_k(self.ids, self.scores, labels, [2])[0]
            if labels.sum() == 0:
                self.assertEqual(row["precision_at_k"], 0)
                self.assertIsNone(row["recall_at_k"])
                self.assertIsNone(row["lift_at_k"])
            else:
                self.assertEqual(row["precision_at_k"], 1)
                self.assertEqual(row["recall_at_k"], 0.4)
                self.assertEqual(row["lift_at_k"], 1)
        self.assertIsNone(random_selection_reference(5, 0, [2])[0]["expected_lift_at_k"])

    def test_invalid_ids_scores_targets_and_capacities_are_rejected(self):
        bad_rankings = [([], []), (["a", "a"], [0.1, 0.2]), ([" "], [0.1]),
                        ([None], [0.1]), (["a"], [np.nan]), (["a"], [np.inf]),
                        (["a"], [-0.1]), (["a"], [1.1]), (["a"], [[0.1]]),
                        (["a", "b"], [0.1])]
        for ids, scores in bad_rankings:
            with self.subTest(ids=ids, scores=scores), self.assertRaises(ValueError):
                rank_customers(ids, scores)
        for values in [[], [0], [-1], [6], [1.5], [True], [2, 2]]:
            with self.subTest(k=values), self.assertRaises(ValueError):
                evaluate_top_k(self.ids, self.scores, self.target, values)
        for target in [[0], [0, 1, 2, 0, 1], [0, 1, np.nan, 0, 1]]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                evaluate_top_k(self.ids, self.scores, target, [1])


if __name__ == "__main__":
    unittest.main()
