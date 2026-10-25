"""Оценить заранее выбранную модель на test; воспроизведение не меняет выбор."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__:
    from .analyze_errors import segment_counts
    from .ranking import evaluate_top_k, random_selection_reference, rank_customers
    from .train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )
    from .tune_models import build_candidate
else:
    from analyze_errors import segment_counts
    from ranking import evaluate_top_k, random_selection_reference, rank_customers
    from train_baseline import (
        DATA_PATH, FEATURES, PROJECT_ROOT, evaluate, load_data,
        save_split_manifest, split_data, write_json,
    )
    from tune_models import build_candidate


SELECTION_COMMIT = "5232d2cfbc898ebc32b3a1b537fa4babf18f100c"
CONFIG_SHA256 = "0449deb599af729dd56d8da0572a24e166e719571fed1afac73f2e8fe63e9a64"
CONFIG_PATH = PROJECT_ROOT / "config" / "model_selection.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "final_test.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path=CONFIG_PATH):
    if sha256(path) != CONFIG_SHA256:
        raise ValueError("Конфигурация отличается от выбора, зафиксированного до test.")
    config = json.loads(Path(path).read_text())
    for name, expected in config["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Версия {name} отличается от зафиксированного окружения.")
    if sha256(PROJECT_ROOT / "scripts" / "train_baseline.py") != config["preprocessing_source_sha256"]:
        raise ValueError("Код разбиения/preprocessing изменился после фиксации протокола.")
    return config


def final_training_data(splits):
    """Объединить только train и validation, проверив непересечение всех частей."""
    ids = {name: set(part["customerID"]) for name, part in splits.items()}
    for name, part in splits.items():
        if len(ids[name]) != len(part):
            raise ValueError(f"Повторяющиеся ID в {name}.")
    if (ids["train"] & ids["validation"] or ids["train"] & ids["test"]
            or ids["validation"] & ids["test"]):
        raise ValueError("Части train/validation/test пересекаются.")
    return pd.concat([splits["train"], splits["validation"]], axis=0)


def fit_models(development, config):
    """Обучение получает только объединённую обучающую часть, без test."""
    models = {}
    for family, spec in config["models"].items():
        model = build_candidate(family, spec["parameters"])
        classifier = model if family == "dummy_prior" else model.named_steps["classifier"]
        if classifier.get_params(deep=False) != spec["classifier_parameters"]:
            raise ValueError(f"Полные параметры {family} отличаются от протокола.")
        print(f"Обучение на train + validation: {family}, {len(development)} клиентов", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(development[FEATURES], development["Churn"])
        models[family] = model
    return models


def bootstrap_intervals(target, probability, reference_probability, *, samples,
                        seed, confidence_level):
    """Парный bootstrap фиксированных прогнозов с сохранением числа классов."""
    target = np.asarray(target)
    probability = np.asarray(probability, dtype=float)
    reference = np.asarray(reference_probability, dtype=float)
    if (target.ndim != 1 or set(target.tolist()) != {0, 1}
            or probability.shape != target.shape or reference.shape != target.shape
            or not np.isfinite(probability).all() or not np.isfinite(reference).all()
            or ((probability < 0) | (probability > 1)).any()
            or ((reference < 0) | (reference > 1)).any()
            or not isinstance(samples, int) or isinstance(samples, bool) or samples < 2
            or not 0 < confidence_level < 1):
        raise ValueError("Bootstrap требует два класса, корректные риски и параметры.")
    groups = [np.flatnonzero(target == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    distributions = {key: [] for key in ("average_precision", "roc_auc", "paired_ap_difference")}
    for _ in range(samples):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        labels, selected, baseline = target[indices], probability[indices], reference[indices]
        ap = float(average_precision_score(labels, selected))
        distributions["average_precision"].append(ap)
        distributions["roc_auc"].append(float(roc_auc_score(labels, selected)))
        distributions["paired_ap_difference"].append(ap - float(average_precision_score(labels, baseline)))
    estimates = {
        "average_precision": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "paired_ap_difference": float(average_precision_score(target, probability)
                                      - average_precision_score(target, reference)),
    }
    alpha = (1 - confidence_level) / 2
    return {key: {"estimate": estimates[key],
                  "lower": float(np.quantile(values, alpha)),
                  "upper": float(np.quantile(values, 1 - alpha))}
            for key, values in distributions.items()}


def campaign_segments(frame, probability, k):
    ranked = rank_customers(frame["customerID"], probability)
    selected = frame["customerID"].isin(ranked.head(k)["customerID"]).to_numpy(dtype=int)
    # Готовый индикатор отбора 0/1: существующий подсчёт >=0.5 сохраняет этот отбор.
    # Это анализ одного глобального top-K, не отдельных top-K внутри сегментов.
    return segment_counts(frame[FEATURES], frame["Churn"], selected)


def assert_reproduced(actual, expected, path="report"):
    """Не заменять опубликованные результаты и не скрывать смену протокола."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Изменились поля {path}.")
        for key in expected:
            assert_reproduced(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Изменился список {path}.")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_reproduced(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        if (not isinstance(actual, (float, int)) or isinstance(actual, bool)
                or not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0)):
            raise ValueError(f"Результат {path} не воспроизвёлся.")
    elif actual != expected:
        raise ValueError(f"Изменилось значение {path}.")


def save_or_verify_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x") as output:
            output.write(contents)
    except FileExistsError:
        assert_reproduced(report, json.loads(path.read_text()))


def print_summary(report):
    print("\nЗафиксированная модель сервиса:", report["selected_family"])
    for family, result in report["models"].items():
        metrics = result["global_metrics"]
        print(f"{family:24} test AP={metrics['average_precision']:.6f}, "
              f"ROC-AUC={metrics['roc_auc']:.6f}")
    for row in report["models"][report["selected_family"]]["top_k"]:
        print(f"K={row['k']:3}: найдено {row['true_positives']}, "
              f"precision={row['precision_at_k']:.4f}, recall={row['recall_at_k']:.4f}")
    print("Результаты test не меняют выбранную модель. Отчёт: reports/final_test.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true",
                        help="Повторить замороженный протокол, сверить отчёт и восстановить локальные модели")
    args = parser.parse_args()
    config = load_protocol()
    if not DATA_PATH.exists() or sha256(DATA_PATH) != config["dataset_sha256"]:
        raise ValueError("Нужен исходный CSV с зафиксированным SHA-256. См. README.md.")
    metadata = {
        "protocol_version": config["protocol_version"], "selection_commit": SELECTION_COMMIT,
        "selection_config_sha256": CONFIG_SHA256, "dataset_sha256": config["dataset_sha256"],
        "selected_family": config["selected_family"], "versions": config["versions"],
    }
    if REPORT_PATH.exists():
        previous = json.loads(REPORT_PATH.read_text())
        for key, value in metadata.items():
            assert_reproduced(previous.get(key), value, key)
        if not args.reproduce:
            print("Финальная оценка уже сохранена. Для проверки и восстановления моделей: --reproduce.")
            print_summary(previous)
            return

    splits = split_data(load_data(DATA_PATH))
    if {name: len(part) for name, part in splits.items()} != config["split_sizes"]:
        raise ValueError("Размеры частей данных изменились.")
    artifacts = PROJECT_ROOT / "artifacts"
    save_split_manifest(artifacts / "splits.json", {
        "dataset_sha256": config["dataset_sha256"], "random_state": config["random_state"],
        "customer_ids": {name: part["customerID"].tolist() for name, part in splits.items()},
    })
    development = final_training_data(splits)
    test = splits["test"]
    save_split_manifest(artifacts / "final_splits.json", {
        "selection_config_sha256": CONFIG_SHA256, "dataset_sha256": config["dataset_sha256"],
        "training_customer_ids": development["customerID"].tolist(),
        "test_customer_ids": test["customerID"].tolist(),
    })
    models = fit_models(development, config)
    results, probabilities = {}, {}
    for family, model in models.items():
        probability = model.predict_proba(test[FEATURES])[:, list(model.classes_).index(1)]
        probabilities[family] = probability
        results[family] = {
            "role": config["models"][family]["role"],
            "parameters": config["models"][family]["parameters"],
            "global_metrics": evaluate(model, test[FEATURES], test["Churn"]),
            "top_k": (None if family == "dummy_prior" else evaluate_top_k(
                test["customerID"], probability, test["Churn"], config["evaluation"]["k_values"])),
        }
    selected = config["selected_family"]
    bootstrap = config["evaluation"]["bootstrap"]
    print("Bootstrap фиксированных прогнозов: 1000 выборок, без переобучения моделей", flush=True)
    uncertainty = bootstrap_intervals(
        test["Churn"], probabilities[selected], probabilities[bootstrap["paired_ap_reference"]],
        samples=bootstrap["samples"], seed=bootstrap["seed"], confidence_level=bootstrap["confidence_level"],
    )
    report = {
        **metadata, "random_state": config["random_state"], "split_sizes": config["split_sizes"],
        "training_splits": config["training_splits"], "training_rows": len(development),
        "training_positives": int(development["Churn"].sum()),
        "evaluation_split": "test", "test_rows": len(test),
        "test_positives": int(test["Churn"].sum()), "test_prevalence": float(test["Churn"].mean()),
        "test_used_for_selection": False, "refit_on_test": False,
        "model_selection_changed_after_test": False,
        "feature_columns": FEATURES, "excluded_columns": ["customerID", "Churn"],
        "evaluation_protocol": config["evaluation"], "service_policy": config["service_policy"],
        "models": results,
        "random_selection_reference": random_selection_reference(
            len(test), int(test["Churn"].sum()), config["evaluation"]["k_values"]),
        "uncertainty": {"protocol": bootstrap, "intervals": uncertainty},
        "selected_model_error_analysis": {
            "classification_threshold_0_5": segment_counts(test[FEATURES], test["Churn"], probabilities[selected]),
            "global_top_k_100": campaign_segments(test, probabilities[selected], config["evaluation"]["campaign_segment_k"]),
        },
        "service_model_path": "artifacts/final/model.joblib",
    }
    save_or_verify_report(REPORT_PATH, report)
    final_dir = artifacts / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for family, model in models.items():
        name = "model.joblib" if family == selected else family + ".joblib"
        path = final_dir / name
        joblib.dump(model, path)
        restored = joblib.load(path)
        restored_probability = restored.predict_proba(test[FEATURES])[:, list(restored.classes_).index(1)]
        np.testing.assert_allclose(restored_probability, probabilities[family], atol=1e-15, rtol=0)
        model_files[name] = sha256(path)
    rank_customers(test["customerID"], probabilities[selected]).to_csv(final_dir / "test_ranking.csv", index=False)
    write_json(final_dir / "manifest.json", {
        **metadata, "training_rows": len(development), "model_sha256": model_files,
        "service_model_file": "model.joblib", "evaluation_report_sha256": sha256(REPORT_PATH),
    })
    print_summary(report)


if __name__ == "__main__":
    main()
