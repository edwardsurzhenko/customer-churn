"""Обучить baseline и логистическую регрессию; оценить только на validation."""

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


def build_logistic_model():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
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


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Сначала запустите: python scripts/download_data.py")
    frame = load_data(DATA_PATH)
    splits = split_data(frame)
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    write_json(artifacts / "splits.json", {
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
    metrics = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train[FEATURES], train["Churn"])
        metrics[name] = evaluate(model, validation[FEATURES], validation["Churn"])

    # Сохраняем весь pipeline, включая преобразования, обученные только на train.
    joblib.dump(models["logistic_regression"], artifacts / "logistic_regression.joblib")
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
    write_json(PROJECT_ROOT / "reports" / "baseline.json", report)

    print("Размеры частей:", report["split_sizes"])
    print("\nValidation, порог 0.5:")
    print(f"{'Модель':24} {'ROC-AUC':>9} {'AP':>9} {'Precision':>10} {'Recall':>9}")
    for name, result in metrics.items():
        print(f"{name:24} {result['roc_auc']:9.4f} {result['average_precision']:9.4f} "
              f"{result['precision']:10.4f} {result['recall']:9.4f}")
    print("\nTest не оценивался. Отчёт: reports/baseline.json")


if __name__ == "__main__":
    main()
