"""Границы финального обучения, неизменность протокола и парная оценка метрик."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.evaluate_final import (
    CONFIG_PATH, bootstrap_intervals, campaign_segments, final_training_data,
    fit_models, load_protocol, save_or_verify_report,
)
from scripts.train_baseline import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from scripts.tune_models import build_candidate


def example_data(size=32):
    frame = pd.DataFrame({name: ["train-category"] * size for name in CATEGORICAL_FEATURES})
    for name in NUMERIC_FEATURES:
        frame[name] = np.arange(size, dtype=float)
    frame.loc[::7, "TotalCharges"] = np.nan
    frame["customerID"] = [f"id-{index:03}" for index in range(size)]
    frame["Churn"] = np.arange(size) % 2
    return frame


class FinalEvaluationTests(unittest.TestCase):
    def test_final_fit_uses_train_and_validation_but_never_test(self):
        frame = example_data()
        frame.loc[20:25, "Contract"] = "validation-category"
        frame.loc[26:, "Contract"] = "test-only-category"
        frame.loc[26:, "TotalCharges"] = 1e9
        parts = {"train": frame.iloc[:20], "validation": frame.iloc[20:26], "test": frame.iloc[26:]}
        development = final_training_data(parts)
        self.assertEqual(development.index.tolist(), list(range(26)))
        original_fit, fit_indices = Pipeline.fit, []

        def checked_fit(model, features, target, **kwargs):
            if "classifier" in model.named_steps:
                fit_indices.append(features.index.tolist())
            return original_fit(model, features, target, **kwargs)

        config = {"models": {}}
        for family, parameters in {
            "gradient_boosting": {"n_estimators": 3, "max_depth": 1, "min_samples_leaf": 1},
            "logistic_regression": {"C": 1.0}, "dummy_prior": {},
        }.items():
            model = build_candidate(family, parameters)
            classifier = model if family == "dummy_prior" else model.named_steps["classifier"]
            config["models"][family] = {"parameters": parameters,
                                         "classifier_parameters": classifier.get_params(deep=False)}
        with patch.object(Pipeline, "fit", checked_fit), contextlib.redirect_stdout(io.StringIO()):
            models = fit_models(development, config)
        self.assertEqual(fit_indices, [list(range(26)), list(range(26))])
        np.testing.assert_allclose(models["dummy_prior"].class_prior_, [0.5, 0.5])
        for family in ("gradient_boosting", "logistic_regression"):
            preprocessing = models[family].named_steps["preprocessing"]
            numeric = preprocessing.named_transformers_["numeric"]
            expected = development[NUMERIC_FEATURES].median()
            np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, expected)
            if "scaler" in numeric.named_steps:
                np.testing.assert_allclose(numeric.named_steps["scaler"].mean_,
                                           development[NUMERIC_FEATURES].fillna(expected).mean())
            categories = preprocessing.named_transformers_["categorical"].named_steps["encoder"].categories_
            learned = set(categories[CATEGORICAL_FEATURES.index("Contract")])
            self.assertEqual(learned, {"train-category", "validation-category"})
            self.assertTrue(np.isfinite(models[family].predict_proba(parts["test"][FEATURES])).all())

    def test_overlapping_test_is_rejected_before_final_fit(self):
        frame = example_data()
        with self.assertRaisesRegex(ValueError, "пересекаются"):
            final_training_data({"train": frame.iloc[:20], "validation": frame.iloc[20:26],
                                 "test": frame.iloc[25:]})

    def test_modified_protocol_is_rejected(self):
        config = json.loads(CONFIG_PATH.read_text())
        config["models"]["gradient_boosting"]["parameters"]["max_depth"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "зафиксированного до test"):
                load_protocol(path)

    def test_bootstrap_is_paired_repeatable_and_handles_perfect_ranking(self):
        target = [0, 0, 0, 1, 1, 1]
        perfect = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]
        kwargs = {"samples": 30, "seed": 42, "confidence_level": 0.95}
        result = bootstrap_intervals(target, perfect, perfect, **kwargs)
        self.assertEqual(result["average_precision"], {"estimate": 1.0, "lower": 1.0, "upper": 1.0})
        self.assertEqual(result["roc_auc"], {"estimate": 1.0, "lower": 1.0, "upper": 1.0})
        self.assertEqual(result["paired_ap_difference"], {"estimate": 0.0, "lower": 0.0, "upper": 0.0})
        imperfect = [0.1, 0.7, 0.3, 0.6, 0.2, 0.9]
        actual = bootstrap_intervals(target, perfect, imperfect, **kwargs)
        self.assertEqual(actual, bootstrap_intervals(target, perfect, imperfect, **kwargs))
        reverse = bootstrap_intervals(target, imperfect, perfect, **kwargs)
        self.assertAlmostEqual(actual["paired_ap_difference"]["lower"], -reverse["paired_ap_difference"]["upper"])
        self.assertAlmostEqual(actual["paired_ap_difference"]["upper"], -reverse["paired_ap_difference"]["lower"])
        with self.assertRaises(ValueError):
            bootstrap_intervals([1, 1], [0.2, 0.7], [0.2, 0.7], **kwargs)

    def test_campaign_segments_account_for_one_global_top_k(self):
        frame = example_data(8)
        frame.loc[:3, "Contract"] = "first"
        frame.loc[4:, "Contract"] = "second"
        result = campaign_segments(frame, np.linspace(0.1, 0.8, 8), 3)
        for segments in result.values():
            self.assertEqual(sum(row["tp"] + row["fp"] for row in segments.values()), 3)
            self.assertEqual(sum(row["tp"] for row in segments.values()), 2)
            self.assertEqual(sum(row["fn"] for row in segments.values()), 2)
        self.assertIsNone(result["Contract"]["first"]["precision"])
        self.assertEqual(result["Contract"]["second"]["precision"], 2 / 3)

    def test_existing_report_is_verified_and_never_overwritten(self):
        report = {"selection_config_sha256": "fixed", "metrics": {"ap": 0.7, "tp": 78}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_or_verify_report(path, report)
            contents = path.read_bytes()
            save_or_verify_report(path, report)
            report["metrics"]["ap"] = 0.7000000000000001
            save_or_verify_report(path, report)
            self.assertEqual(path.read_bytes(), contents)
            report["metrics"]["ap"] = 0.8
            with self.assertRaisesRegex(ValueError, "не воспроизвёлся"):
                save_or_verify_report(path, report)
            self.assertEqual(path.read_bytes(), contents)


if __name__ == "__main__":
    unittest.main()
