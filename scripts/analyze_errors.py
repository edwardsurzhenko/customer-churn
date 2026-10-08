"""Исследовать train и проверить одну гипотезу на фиксированной validation."""

import hashlib
from importlib.metadata import version
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

if __package__:
    from .train_baseline import (
        DATA_PATH, FEATURES, NUMERIC_FEATURES, PROJECT_ROOT, RANDOM_STATE,
        build_boosting_model, build_forest_model, build_logistic_model,
        evaluate, load_data, save_split_manifest, split_data, write_json,
    )
else:
    from train_baseline import (
        DATA_PATH, FEATURES, NUMERIC_FEATURES, PROJECT_ROOT, RANDOM_STATE,
        build_boosting_model, build_forest_model, build_logistic_model,
        evaluate, load_data, save_split_manifest, split_data, write_json,
    )


def zero_new_customer_charges(features):
    """Проверяемая гипотеза: пропуск TotalCharges при tenure=0 заменить нулём."""
    result = features.copy()
    mask = result["tenure"].eq(0) & result["TotalCharges"].isna()
    result.loc[mask, "TotalCharges"] = 0.0
    return result


def tenure_groups(features):
    return pd.cut(
        features["tenure"], bins=[-np.inf, 12, 24, 48, np.inf],
        labels=["0–12", "13–24", "25–48", "49+"],
    ).astype(object).fillna("missing")


def segment_counts(features, target, probability=None):
    """Агрегаты по заранее заданным группам; нулевой знаменатель обозначаем None."""
    rows = pd.DataFrame({"target": np.asarray(target)}, index=features.index)
    rows["Contract"] = features["Contract"].fillna("missing")
    rows["tenure_group"] = tenure_groups(features)
    if probability is not None:
        predicted = np.asarray(probability) >= 0.5
        actual = rows["target"].to_numpy() == 1
        rows["tp"] = actual & predicted
        rows["fp"] = ~actual & predicted
        rows["fn"] = actual & ~predicted
        rows["tn"] = ~actual & ~predicted
    segments = {}
    for column in ["Contract", "tenure_group"]:
        segments[column] = {}
        for label, group in rows.groupby(column, sort=True):
            count, positives = len(group), int(group["target"].sum())
            result = {"count": count, "positives": positives,
                      "churn_rate": positives / count}
            if probability is not None:
                result.update({name: int(group[name].sum())
                               for name in ["tp", "fp", "fn", "tn"]})
                selected = result["tp"] + result["fp"]
                result["precision"] = result["tp"] / selected if selected else None
                result["recall"] = result["tp"] / positives if positives else None
            segments[column][str(label)] = result
    return segments


def missing_charge_profile(features):
    missing = features["TotalCharges"].isna()
    return {
        "missing_count": int(missing.sum()),
        "missing_with_zero_tenure": int((missing & features["tenure"].eq(0)).sum()),
        "missing_other_or_unknown_tenure": int((missing & ~features["tenure"].eq(0)).sum()),
    }


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Сначала запустите: python scripts/download_data.py")
    frame = load_data(DATA_PATH)
    splits = split_data(frame)
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    save_split_manifest(PROJECT_ROOT / "artifacts" / "splits.json", {
        "dataset_sha256": data_hash, "random_state": RANDOM_STATE,
        "customer_ids": {name: part["customerID"].tolist()
                         for name, part in splits.items()},
    })
    train, validation = splits["train"], splits["validation"]
    factories = {
        "logistic_regression": build_logistic_model,
        "random_forest": build_forest_model,
        "gradient_boosting": build_boosting_model,
    }
    experiments = {}
    for variant in ["median", "zero_for_new_customers"]:
        experiments[variant] = {}
        for name, factory in factories.items():
            print(f"Обучение: {variant} / {name}", flush=True)
            model = factory()
            if variant == "zero_for_new_customers":
                model = Pipeline([
                    ("new_customer_rule", FunctionTransformer(zero_new_customer_charges)),
                    ("model", model),
                ])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train[FEATURES], train["Churn"])
            probability = model.predict_proba(validation[FEATURES])[
                :, list(model.classes_).index(1)
            ]
            experiments[variant][name] = {
                "metrics": evaluate(model, validation[FEATURES], validation["Churn"]),
                "segments": segment_counts(validation[FEATURES], validation["Churn"],
                                           probability),
            }
    report = {
        "dataset_sha256": data_hash, "random_state": RANDOM_STATE,
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "evaluation_split": "validation", "test_evaluated": False,
        "positive_class": "Churn=Yes", "classification_threshold": 0.5,
        "feature_columns": FEATURES, "excluded_columns": ["customerID", "Churn"],
        "versions": {name: version(name)
                     for name in ["pandas", "numpy", "scikit-learn", "joblib"]},
        "train_profile": {
            "count": len(train), "positives": int(train["Churn"].sum()),
            "churn_rate": float(train["Churn"].mean()),
            "numeric_missing_counts": {
                name: int(train[name].isna().sum()) for name in NUMERIC_FEATURES
            },
            "total_charges": {
                **missing_charge_profile(train[FEATURES]),
                "observed_median": float(train["TotalCharges"].median()),
            },
            "segments": segment_counts(train[FEATURES], train["Churn"]),
        },
        "validation_total_charges": missing_charge_profile(validation[FEATURES]),
        "hypothesis": "Missing TotalCharges with tenure=0 can be imputed with zero.",
        "variants": {
            "median": "Unchanged baseline preprocessing and model settings.",
            "zero_for_new_customers": (
                "Fill only missing TotalCharges when tenure=0 with zero; "
                "otherwise use the unchanged baseline pipeline."
            ),
        },
        "experiments": experiments,
        "final_model_selected": False,
        "production_preprocessing_changed": False,
    }
    write_json(PROJECT_ROOT / "reports" / "error_analysis.json", report)
    print("\nValidation, порог 0.5:")
    print(f"{'Модель':24} {'AP median':>10} {'AP zero':>10} {'Разница':>10}")
    for name in factories:
        baseline = experiments["median"][name]["metrics"]["average_precision"]
        candidate = experiments["zero_for_new_customers"][name]["metrics"]["average_precision"]
        print(f"{name:24} {baseline:10.6f} {candidate:10.6f} "
              f"{candidate - baseline:+10.6f}")
    print("\nTest не оценивался. Исходные модели и отчёты не заменены.")
    print("Отчёт: reports/error_analysis.json")


if __name__ == "__main__":
    main()
