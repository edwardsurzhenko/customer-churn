"""Обучить первые модели и ансамбли; оценить только на validation."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
RANDOM_STATE = 42
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path):
    """Прочитать таблицу и преобразовать типы без обучения на данных."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = set(FEATURES + ["customerID", "Churn"])
    if set(frame.columns) != expected or frame.empty:
        raise ValueError("CSV должен содержать 21 ожидаемый столбец и непустые данные.")
    frame = frame.apply(lambda column: column.str.strip())
    if frame["customerID"].eq("").any() or frame["customerID"].duplicated().any():
        raise ValueError("customerID должен быть непустым и уникальным.")
    if set(frame["Churn"]) != {"Yes", "No"}:
        raise ValueError("Churn должен содержать только Yes и No, оба класса обязательны.")

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Бесконечное значение в {column}.")
    frame[CATEGORICAL_FEATURES] = (
        frame[CATEGORICAL_FEATURES].astype(object).replace("", np.nan)
    )
    frame["Churn"] = frame["Churn"].map({"No": 0, "Yes": 1})
    return frame


def split_data(frame):
    """Разделить клиентов на train/validation/test с сохранением долей классов."""
    train_validation, test = train_test_split(
        frame, test_size=0.20, stratify=frame["Churn"], random_state=RANDOM_STATE,
    )
    train, validation = train_test_split(
        train_validation, test_size=0.25, stratify=train_validation["Churn"],
        random_state=RANDOM_STATE,
    )
    return {"train": train, "validation": validation, "test": test}


def build_preprocessing(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")


def build_logistic_model():
    return Pipeline([
        ("preprocessing", build_preprocessing(scale_numeric=True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_forest_model():
    return Pipeline([
        ("preprocessing", build_preprocessing(scale_numeric=False)),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            max_features="sqrt", bootstrap=True, n_jobs=2, random_state=RANDOM_STATE,
        )),
    ])


def build_boosting_model():
    return Pipeline([
        ("preprocessing", build_preprocessing(scale_numeric=False)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            min_samples_leaf=10, random_state=RANDOM_STATE,
        )),
    ])


def evaluate(model, features, target):
    probability = model.predict_proba(features)[:, list(model.classes_).index(1)]
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
    }


def write_json(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, ensure_ascii=False, indent=2) + "\n")


def save_split_manifest(path, manifest):
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Данные или разбиение отличаются от artifacts/splits.json. "
                             "Нельзя незаметно менять выборку для сравнения моделей.")
        return
    write_json(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-models", action="store_true",
                        help="Сравнить также случайный лес и градиентный бустинг")
    args = parser.parse_args()
    if not DATA_PATH.exists():
        raise SystemExit("Сначала запустите: python scripts/download_data.py")
    frame = load_data(DATA_PATH)
    splits = split_data(frame)
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    save_split_manifest(artifacts / "splits.json", {
        "dataset_sha256": data_hash,
        "random_state": RANDOM_STATE,
        "customer_ids": {
            name: part["customerID"].tolist() for name, part in splits.items()
        },
    })

    train, validation = splits["train"], splits["validation"]
    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": build_logistic_model(),
    }
    if args.compare_models:
        models.update({
            "random_forest": build_forest_model(),
            "gradient_boosting": build_boosting_model(),
        })
    metrics = {}
    model_parameters = {}
    tree_depths = {}
    for name, model in models.items():
        print(f"Обучение: {name}", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train[FEATURES], train["Churn"])
        metrics[name] = evaluate(model, validation[FEATURES], validation["Churn"])
        classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
        model_parameters[name] = classifier.get_params(deep=False)
        if name in {"random_forest", "gradient_boosting"}:
            trees = np.asarray(classifier.estimators_, dtype=object).reshape(-1)
            depths = [tree.get_depth() for tree in trees]
            tree_depths[name] = {
                "count": len(depths), "min": min(depths), "max": max(depths),
                "median": float(np.median(depths)),
            }
        # Сохраняем pipeline с преобразованиями, обученными только на train.
        joblib.dump(model, artifacts / f"{name}.joblib")
    report = {
        "dataset_sha256": data_hash,
        "random_state": RANDOM_STATE,
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "evaluation_split": "validation",
        "test_evaluated": False,
        "positive_class": "Churn=Yes",
        "classification_threshold": 0.5,
        "feature_columns": FEATURES,
        "excluded_columns": ["customerID", "Churn"],
        "versions": {
            name: version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib"]
        },
        "metrics": metrics,
    }
    if args.compare_models:
        report.update({
            "model_parameters": model_parameters,
            "tree_depths": tree_depths,
            "comparison_metric": "average_precision",
            "ranking_by_validation_ap": sorted(
                metrics, key=lambda name: metrics[name]["average_precision"], reverse=True,
            ),
            "final_model_selected": False,
        })
    report_name = "comparison.json" if args.compare_models else "baseline.json"
    write_json(PROJECT_ROOT / "reports" / report_name, report)

    print("Размеры частей:", report["split_sizes"])
    print("\nValidation, порог 0.5:")
    print(f"{'Модель':24} {'ROC-AUC':>9} {'AP':>9} {'Precision':>10} {'Recall':>9}")
    for name, result in metrics.items():
        print(f"{name:24} {result['roc_auc']:9.4f} {result['average_precision']:9.4f} "
              f"{result['precision']:10.4f} {result['recall']:9.4f}")
    print(f"\nTest не оценивался. Отчёт: reports/{report_name}")


if __name__ == "__main__":
    main()
