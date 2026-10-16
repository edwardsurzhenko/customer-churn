"""Расширенный случайный поиск на train; сравнение выбранных моделей на validation."""

import hashlib
from importlib.metadata import version
import json
import warnings

import joblib
import numpy as np
from scipy.stats import loguniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV

if __package__:
    from .evaluate_campaign import K_VALUES, check_reproduced_metrics, check_tuning_report
    from .ranking import evaluate_top_k, random_selection_reference, rank_customers
    from .train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, RANDOM_STATE, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )
    from .tune_models import CV_FOLDS, CV_SEEDS, build_candidate, make_cv_splits, run_cv, summarize_scores
else:
    from evaluate_campaign import K_VALUES, check_reproduced_metrics, check_tuning_report
    from ranking import evaluate_top_k, random_selection_reference, rank_customers
    from train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, RANDOM_STATE, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )
    from tune_models import CV_FOLDS, CV_SEEDS, build_candidate, make_cv_splits, run_cv, summarize_scores


# Пространства и бюджет фиксируются до запуска, без подстройки по validation.
SEARCH_SEED = 42
SEARCH_BUDGETS = {"logistic_regression": 16, "random_forest": 24, "gradient_boosting": 24}
SEARCH_SPACES = {
    "logistic_regression": {
        "C": {"distribution": "loguniform", "low": 0.001, "high": 1000.0},
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8, 12, None],
        "min_samples_leaf": [1, 3, 5, 10, 20],
        "max_features": ["sqrt", 0.5, 1.0],
        "class_weight": [None, "balanced"],
    },
    "gradient_boosting": {
        "learning_rate": {"distribution": "loguniform", "low": 0.01, "high": 0.2},
        "n_estimators": [100, 150, 300, 500],
        "max_depth": [1, 2, 3, 4],
        "min_samples_leaf": [5, 10, 20, 40],
        "subsample": [0.7, 1.0],
    },
}
METRICS = ("average_precision", "roc_auc")


def parameter_distributions(family):
    return {
        "classifier__" + name: (loguniform(spec["low"], spec["high"])
                               if isinstance(spec, dict) else spec)
        for name, spec in SEARCH_SPACES[family].items()
    }


def search_candidates(family, features, target, folds, *, distributions=None,
                      n_iter=None, n_jobs=2, verbose=2):
    """Получает только train; возвращает оценки CV без refit на всей выборке."""
    search = RandomizedSearchCV(
        build_candidate(family, {}),
        parameter_distributions(family) if distributions is None else distributions,
        n_iter=SEARCH_BUDGETS[family] if n_iter is None else n_iter,
        scoring={metric: metric for metric in METRICS}, cv=folds,
        refit=False, random_state=SEARCH_SEED, n_jobs=n_jobs, pre_dispatch=n_jobs,
        error_score="raise", return_train_score=False, verbose=verbose,
    )
    # Два параллельных обучения; лес внутри одного обучения использует два потока.
    # Threading также сохраняет общий фильтр ConvergenceWarning для всех fit.
    with warnings.catch_warnings(), joblib.parallel_config(backend="threading"):
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(features, target)
    results = search.cv_results_
    candidates = []
    for index, parameters in enumerate(results["params"]):
        candidate = {
            "source": "random_search",
            "parameters": {
                name.removeprefix("classifier__"): (value.item() if isinstance(value, np.generic) else value)
                for name, value in parameters.items()
            },
        }
        for metric in METRICS:
            candidate[metric] = summarize_scores([
                results[f"split{fold}_test_{metric}"][index] for fold in range(len(folds))
            ])
        candidates.append(candidate)
    return candidates


def select_candidate(candidates):
    """Максимум средней AP CV; при равенстве сохраняется первый, прежний вариант."""
    scores = [candidate["average_precision"]["mean"] for candidate in candidates]
    if not scores or not np.isfinite(scores).all():
        raise ValueError("Нужны непустые и конечные оценки AP CV.")
    return max(range(len(candidates)), key=lambda index: scores[index])


def main():
    tuning_path = PROJECT_ROOT / "reports" / "tuning.json"
    if not DATA_PATH.exists() or not tuning_path.exists():
        raise SystemExit("Нужны исходный CSV и reports/tuning.json. См. README.md.")
    tuning = json.loads(tuning_path.read_text())
    splits = split_data(load_data(DATA_PATH))
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    split_sizes = {name: len(part) for name, part in splits.items()}
    check_tuning_report(tuning, data_hash, split_sizes)
    if (tuning["cv"]["seeds"] != list(CV_SEEDS)
            or tuning["cv"]["folds_per_repeat"] != CV_FOLDS
            or tuning["cv"]["preprocessing_fitted_inside_each_fold"] is not True
            or any(version(name) != expected for name, expected in tuning["versions"].items())):
        raise ValueError("CV или версии библиотек отличаются от предыдущего эксперимента.")
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
        "held_out_customer_ids": [train.iloc[held_out]["customerID"].tolist() for _, held_out in folds],
    })

    candidates, selected_indices = {}, {}
    for family, budget in SEARCH_BUDGETS.items():
        previous = tuning["validation_results"][family]
        parameters = previous["selected_parameters"]
        print(f"\nКонтроль прежней конфигурации: {family}", flush=True)
        scores = run_cv(build_candidate(family, parameters), train[FEATURES], train["Churn"], folds)
        control = {"source": "previous_cv_winner", "parameters": parameters}
        old = tuning["candidates"][family][previous["selected_candidate_index"]]
        for metric in METRICS:
            control[metric] = summarize_scores(scores[f"test_{metric}"])
            if not np.allclose(control[metric]["fold_scores"], old[metric]["fold_scores"],
                               atol=1e-12, rtol=0):
                raise ValueError(f"Прежние оценки CV {family}/{metric} не воспроизвелись.")
        print(f"Случайный поиск: {family}, {budget} сочетаний × {len(folds)} фолдов", flush=True)
        candidates[family] = [control] + search_candidates(
            family, train[FEATURES], train["Churn"], folds,
        )
        selected_indices[family] = select_candidate(candidates[family])
        winner = candidates[family][selected_indices[family]]
        print(f"Выбрано по CV: AP={winner['average_precision']['mean']:.6f}, "
              f"{winner['parameters']}", flush=True)
        write_json(artifacts / "expanded_search_cv.json", {
            "dataset_sha256": data_hash, "candidates": candidates,
            "selected_indices": selected_indices, "validation_evaluated": False,
        })

    # Сначала заканчивается весь поиск и фиксируются три выбора по CV.
    # Только после этого модели переобучаются на всём train и проверяются на validation.
    rankings = artifacts / "rankings"
    rankings.mkdir(parents=True, exist_ok=True)
    validation_results = {}
    for family, selected_index in selected_indices.items():
        results = {}
        for index in dict.fromkeys([0, selected_index]):
            parameters = candidates[family][index]["parameters"]
            print(f"Train → validation: {family}, кандидат {index}", flush=True)
            model = build_candidate(family, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train[FEATURES], train["Churn"])
            metrics = evaluate(model, validation[FEATURES], validation["Churn"])
            if index == 0:
                check_reproduced_metrics(metrics, tuning["validation_results"][family]["selected_metrics"])
            probability = model.predict_proba(validation[FEATURES])[:, list(model.classes_).index(1)]
            results[index] = {
                "parameters": parameters, "global_metrics": metrics,
                "top_k": evaluate_top_k(validation["customerID"], probability,
                                        validation["Churn"], K_VALUES),
            }
            variant = "selected" if index == selected_index else "previous"
            rank_customers(validation["customerID"], probability).to_csv(
                rankings / f"expanded_{family}_{variant}_validation.csv", index=False,
            )
            if index == selected_index:
                joblib.dump(model, artifacts / f"expanded_selected_{family}.joblib")
        validation_results[family] = {
            "selected_candidate_index": selected_index,
            "previous": results[0], "selected": results[selected_index],
        }
    positives = int(validation["Churn"].sum())
    report = {
        "dataset_sha256": data_hash,
        "tuning_report_sha256": hashlib.sha256(tuning_path.read_bytes()).hexdigest(),
        "random_state": RANDOM_STATE, "split_sizes": split_sizes,
        "feature_columns": FEATURES, "excluded_columns": ["customerID", "Churn"],
        "positive_class": "Churn=Yes", "preprocessing": tuning["preprocessing"],
        "baseline_classifier_parameters": tuning["baseline_classifier_parameters"],
        "search": {
            "method": "RandomizedSearchCV", "seed": SEARCH_SEED,
            "budgets": SEARCH_BUDGETS, "spaces": SEARCH_SPACES,
            "previous_cv_winners_included": True, "refit_during_search": False,
            "cv_fits": (sum(SEARCH_BUDGETS.values()) + len(SEARCH_BUDGETS)) * len(folds),
            "n_jobs": 2, "backend": "threading",
        },
        "cv": {**tuning["cv"], "tie_break": "previous winner first, then sampling order"},
        "validation_used_for_parameter_selection": False,
        "evaluation_split": "validation", "test_evaluated": False,
        "final_model_selected": False, "campaign_size_selected": False,
        "classification_threshold": 0.5, "probability_threshold_for_top_k": None,
        "ranking_rule": "risk_probability descending, customerID ascending for exact ties",
        "ranking_uses_target": False, "k_values": list(K_VALUES),
        "validation_positives": positives,
        "validation_prevalence": positives / len(validation),
        "random_selection_reference": random_selection_reference(len(validation), positives, K_VALUES),
        "dummy_prior_metrics": tuning["validation_results"]["dummy_prior"]["selected_metrics"],
        "candidates": candidates, "validation_results": validation_results,
        "versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "scipy", "joblib"]},
    }
    write_json(PROJECT_ROOT / "reports" / "expanded_search.json", report)
    print("\nValidation: прежняя AP → AP после поиска; прежний TP@100 → новый TP@100")
    for family, result in validation_results.items():
        old, new = result["previous"], result["selected"]
        old_tp = next(row["true_positives"] for row in old["top_k"] if row["k"] == 100)
        new_tp = next(row["true_positives"] for row in new["top_k"] if row["k"] == 100)
        print(f"{family:24} {old['global_metrics']['average_precision']:.6f} → "
              f"{new['global_metrics']['average_precision']:.6f}; {old_tp} → {new_tp}")
    print("Test не оценивался. Отчёт: reports/expanded_search.json")


if __name__ == "__main__":
    main()
