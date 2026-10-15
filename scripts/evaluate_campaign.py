"""Оценить списки K клиентов на validation для исходных и выбранных по CV моделей."""

import hashlib
from importlib.metadata import version
import json
import math
import warnings

from sklearn.exceptions import ConvergenceWarning

if __package__:
    from .ranking import evaluate_top_k, random_selection_reference, rank_customers
    from .train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, RANDOM_STATE, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )
    from .tune_models import FACTORIES, PARAMETER_OPTIONS, build_candidate
else:
    from ranking import evaluate_top_k, random_selection_reference, rank_customers
    from train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, RANDOM_STATE, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )
    from tune_models import FACTORIES, PARAMETER_OPTIONS, build_candidate


# Размеры сценариев заданы до расчёта метрик; окончательный бюджет кампании не выбран.
K_VALUES = (50, 100, 200, 500)


def check_tuning_report(report, data_hash, split_sizes):
    """Не переносить настройки из отчёта с другими данными или протоколом."""
    if (report["dataset_sha256"] != data_hash or report["random_state"] != RANDOM_STATE
            or report["split_sizes"] != split_sizes or report["feature_columns"] != FEATURES
            or report["cv"]["data_split"] != "train"
            or report["validation_used_for_parameter_selection"] is not False):
        raise ValueError("Отчёт tuning.json не соответствует текущим данным или протоколу.")
    for family in FACTORIES:
        baseline = build_candidate(family, {})
        classifier = baseline if family == "dummy_prior" else baseline.named_steps["classifier"]
        if classifier.get_params(deep=False) != report["baseline_classifier_parameters"][family]:
            raise ValueError(f"Исходные настройки {family} изменились после CV.")
        candidates = report["candidates"][family]
        if [candidate["parameters"] for candidate in candidates] != PARAMETER_OPTIONS[family]:
            raise ValueError(f"Сетка {family} изменилась после CV.")
        selected = report["validation_results"][family]
        best = max(range(len(candidates)),
                   key=lambda index: candidates[index]["average_precision"]["mean"])
        if (selected["selected_candidate_index"] != best
                or selected["selected_parameters"] != candidates[best]["parameters"]):
            raise ValueError(f"Выбор {family} не соответствует максимуму средней AP CV.")


def check_reproduced_metrics(actual, expected):
    for name, value in actual.items():
        matches = (value == expected[name] if isinstance(value, list)
                   else math.isclose(value, expected[name], abs_tol=1e-12, rel_tol=0))
        if not matches:
            raise ValueError(f"Метрика {name} отличается от отчёта CV. Проверьте окружение.")


def main():
    tuning_path = PROJECT_ROOT / "reports" / "tuning.json"
    if not DATA_PATH.exists() or not tuning_path.exists():
        raise SystemExit("Нужны исходный CSV и reports/tuning.json. См. README.md.")
    splits = split_data(load_data(DATA_PATH))
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    split_sizes = {name: len(part) for name, part in splits.items()}
    tuning = json.loads(tuning_path.read_text())
    check_tuning_report(tuning, data_hash, split_sizes)
    save_split_manifest(PROJECT_ROOT / "artifacts" / "splits.json", {
        "dataset_sha256": data_hash, "random_state": RANDOM_STATE,
        "customer_ids": {name: part["customerID"].tolist() for name, part in splits.items()},
    })
    train, validation = splits["train"], splits["validation"]
    rankings_dir = PROJECT_ROOT / "artifacts" / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    dummy_metrics = None
    for family in FACTORIES:
        selected = tuning["validation_results"][family]
        variants = {"baseline": PARAMETER_OPTIONS[family][0],
                    "cv_selected": selected["selected_parameters"]}
        cache = {}
        models[family] = {}
        for variant, parameters in variants.items():
            cache_key = json.dumps(parameters, sort_keys=True)
            if cache_key not in cache:
                print(f"Train → validation: {family} / {variant}", flush=True)
                model = build_candidate(family, parameters)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(train[FEATURES], train["Churn"])
                probability = model.predict_proba(validation[FEATURES])[:, list(model.classes_).index(1)]
                metrics = evaluate(model, validation[FEATURES], validation["Churn"])
                cache[cache_key] = (probability, metrics)
            probability, metrics = cache[cache_key]
            expected_key = "baseline_metrics" if variant == "baseline" else "selected_metrics"
            check_reproduced_metrics(metrics, selected[expected_key])
            if family == "dummy_prior":
                dummy_metrics = metrics
                continue
            models[family][variant] = {
                "parameters": parameters, "global_metrics": metrics,
                "top_k": evaluate_top_k(validation["customerID"], probability,
                                        validation["Churn"], K_VALUES),
            }
            rank_customers(validation["customerID"], probability).to_csv(
                rankings_dir / f"{family}_{variant}_validation.csv", index=False,
            )
    del models["dummy_prior"]
    positives = int(validation["Churn"].sum())
    report = {
        "dataset_sha256": data_hash, "random_state": RANDOM_STATE,
        "tuning_report_sha256": hashlib.sha256(tuning_path.read_bytes()).hexdigest(),
        "split_sizes": split_sizes, "training_split": "train", "evaluation_split": "validation",
        "test_evaluated": False, "final_model_selected": False, "campaign_size_selected": False,
        "k_values": list(K_VALUES), "validation_positives": positives,
        "validation_prevalence": positives / len(validation),
        "positive_class": "Churn=Yes", "feature_columns": FEATURES,
        "excluded_columns": ["customerID", "Churn"],
        "ranking_rule": "risk_probability descending, customerID ascending for exact ties",
        "ranking_uses_target": False, "probability_threshold_for_top_k": None,
        "global_classification_threshold": 0.5,
        "models": models,
        "dummy_prior": {"global_metrics": dummy_metrics,
                        "ranking": "All scores equal; use the random-selection expectation as reference."},
        "random_selection_reference": random_selection_reference(len(validation), positives, K_VALUES),
        "versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib"]},
    }
    write_json(PROJECT_ROOT / "reports" / "campaign.json", report)
    print("\nValidation: выбранные по CV конфигурации")
    print(f"{'Модель':24} {'K':>4} {'Найдено':>8} {'Precision@K':>12} {'Recall@K':>10} {'Lift@K':>8}")
    for family, variants in models.items():
        for row in variants["cv_selected"]["top_k"]:
            print(f"{family:24} {row['k']:4} {row['true_positives']:8} "
                  f"{row['precision_at_k']:12.4f} {row['recall_at_k']:10.4f} {row['lift_at_k']:8.4f}")
    print("Test не оценивался. Отчёт: reports/campaign.json")


if __name__ == "__main__":
    main()
