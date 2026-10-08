"""Сравнить настройки через CV только на train; оценить их на validation."""

import hashlib
from importlib.metadata import version
import warnings

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_validate

if __package__:
    from .train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, RANDOM_STATE, build_boosting_model,
        build_forest_model, build_logistic_model, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )
else:
    from train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, RANDOM_STATE, build_boosting_model,
        build_forest_model, build_logistic_model, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )


CV_SEEDS = (42, 43)
CV_FOLDS = 5
FACTORIES = {
    "dummy_prior": lambda: DummyClassifier(strategy="prior"),
    "logistic_regression": build_logistic_model,
    "random_forest": build_forest_model,
    "gradient_boosting": build_boosting_model,
}
# Первый вариант каждой семьи — исходные настройки. Сетка задана до запуска CV.
PARAMETER_OPTIONS = {
    "dummy_prior": [{}],
    "logistic_regression": [{"C": 1.0}, {"C": 0.1}, {"C": 10.0}],
    "random_forest": [
        {"max_depth": 8, "min_samples_leaf": 5},
        {"max_depth": 6, "min_samples_leaf": 5},
        {"max_depth": 8, "min_samples_leaf": 10},
        {"max_depth": 12, "min_samples_leaf": 5},
    ],
    "gradient_boosting": [
        {"max_depth": 2, "n_estimators": 150},
        {"max_depth": 1, "n_estimators": 150},
        {"max_depth": 2, "n_estimators": 300},
        {"max_depth": 3, "n_estimators": 150},
    ],
}


def make_cv_splits(target, seeds=CV_SEEDS, n_splits=CV_FOLDS):
    """Одинаковые стратифицированные фолды для всех вариантов настроек."""
    return [
        (fit_indices, held_out_indices)
        for seed in seeds
        for fit_indices, held_out_indices in StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed,
        ).split(np.zeros(len(target)), target)
    ]


def build_candidate(family, parameters):
    model = FACTORIES[family]()
    prefix = "" if family == "dummy_prior" else "classifier__"
    return model.set_params(**{prefix + key: value for key, value in parameters.items()})


def run_cv(model, features, target, folds, **kwargs):
    """Передаём весь pipeline: преобразования заново обучаются внутри фолда."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return cross_validate(
            model, features, target, cv=folds, n_jobs=1, error_score="raise",
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
            **kwargs,
        )


def summarize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    return {
        "fold_scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "repeat_means": [float(part.mean()) for part in scores.reshape(len(CV_SEEDS), CV_FOLDS)],
    }


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Сначала запустите: python scripts/download_data.py")
    splits = split_data(load_data(DATA_PATH))
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    artifacts = PROJECT_ROOT / "artifacts"
    save_split_manifest(artifacts / "splits.json", {
        "dataset_sha256": data_hash, "random_state": RANDOM_STATE,
        "customer_ids": {name: part["customerID"].tolist() for name, part in splits.items()},
    })
    train, validation = splits["train"], splits["validation"]
    folds = make_cv_splits(train["Churn"])
    save_split_manifest(artifacts / "cv_splits.json", {
        "dataset_sha256": data_hash, "seeds": list(CV_SEEDS), "n_splits": CV_FOLDS,
        "train_customer_ids": train["customerID"].tolist(),
        "held_out_customer_ids": [train.iloc[held_out]["customerID"].tolist()
                                  for _, held_out in folds],
    })
    candidates = {}
    selected_indices = {}
    # Validation не участвует в этом цикле и в выборе настроек каждой семьи.
    for family, options in PARAMETER_OPTIONS.items():
        candidates[family] = []
        for index, parameters in enumerate(options):
            print(f"CV: {family} {index + 1}/{len(options)} {parameters}", flush=True)
            scores = run_cv(build_candidate(family, parameters), train[FEATURES],
                            train["Churn"], folds)
            result = {"parameters": parameters, "baseline": index == 0}
            for metric in ["average_precision", "roc_auc"]:
                result[metric] = summarize_scores(scores[f"test_{metric}"])
            candidates[family].append(result)
            print(f"  AP: {result['average_precision']['mean']:.6f}", flush=True)
        selected_indices[family] = max(
            range(len(options)),
            key=lambda index: candidates[family][index]["average_precision"]["mean"],
        )

    validation_results = {}
    for family, selected_index in selected_indices.items():
        results = {}
        # Исходная конфигурация оценивается рядом с выбранной, без замены старых отчётов.
        for index in dict.fromkeys([0, selected_index]):
            model = build_candidate(family, PARAMETER_OPTIONS[family][index])
            print(f"Train → validation: {family}, вариант {index}", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train[FEATURES], train["Churn"])
            results[index] = evaluate(model, validation[FEATURES], validation["Churn"])
            if index == selected_index:
                joblib.dump(model, artifacts / f"cv_selected_{family}.joblib")
        validation_results[family] = {
            "selected_candidate_index": selected_index,
            "selected_parameters": PARAMETER_OPTIONS[family][selected_index],
            "baseline_metrics": results[0], "selected_metrics": results[selected_index],
        }
    report = {
        "dataset_sha256": data_hash, "random_state": RANDOM_STATE,
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "cv": {"data_split": "train", "folds_per_repeat": CV_FOLDS,
               "seeds": list(CV_SEEDS), "shared_folds": True,
               "preprocessing_fitted_inside_each_fold": True,
               "selection_metric": "mean_average_precision",
               "tie_break": "first candidate in declared order",
               "std_is_confidence_interval": False},
        "evaluation_split": "validation", "classification_threshold": 0.5,
        "validation_used_for_parameter_selection": False,
        "test_evaluated": False, "final_model_selected": False,
        "positive_class": "Churn=Yes", "feature_columns": FEATURES,
        "excluded_columns": ["customerID", "Churn"],
        "preprocessing": "Unchanged median baseline; no zero-tenure rule.",
        "baseline_classifier_parameters": {
            family: (model.get_params(deep=False) if family == "dummy_prior"
                     else model.named_steps["classifier"].get_params(deep=False))
            for family in FACTORIES for model in [FACTORIES[family]()]
        },
        "versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib"]},
        "candidates": candidates, "validation_results": validation_results,
    }
    write_json(PROJECT_ROOT / "reports" / "tuning.json", report)
    print("\nValidation AP: исходная → выбранная по CV")
    for family, result in validation_results.items():
        print(f"{family:24} {result['baseline_metrics']['average_precision']:.6f} → "
              f"{result['selected_metrics']['average_precision']:.6f}")
    print("Test не оценивался. Отчёт: reports/tuning.json")


if __name__ == "__main__":
    main()
