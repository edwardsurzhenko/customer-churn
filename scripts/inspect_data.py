"""Первое знакомство с CSV: размеры, целевая переменная и пустые значения."""

from collections import Counter
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Данные ещё не скачаны. Запустите: python scripts/download_data.py")

    with DATA_PATH.open(encoding="utf-8-sig", newline="") as source:
        table = csv.DictReader(source)
        columns = table.fieldnames or []
        rows = list(table)

    if not rows or not {"customerID", "Churn"}.issubset(columns):
        raise SystemExit("CSV пуст или не содержит customerID и Churn.")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise SystemExit("В CSV есть строки с неверным количеством полей.")

    labels = Counter(row["Churn"] for row in rows)
    if set(labels) != {"Yes", "No"}:
        raise SystemExit(f"Неожиданные значения Churn: {dict(labels)}")

    print(f"Клиентов: {len(rows)}")
    print(f"Столбцов: {len(columns)}")
    print(f"Churn = Yes: {labels['Yes']} ({labels['Yes'] / len(rows):.2%})")
    print(f"Churn = No: {labels['No']} ({labels['No'] / len(rows):.2%})")
    repeated_ids = len(rows) - len({row["customerID"] for row in rows})
    print(f"Повторных customerID: {repeated_ids}")

    print("\nПустые значения (включая строки из пробелов):")
    missing = {column: sum(not row[column].strip() for row in rows) for column in columns}
    for column, count in missing.items():
        if count:
            print(f"  {column}: {count}")
    if not any(missing.values()):
        print("  Не найдены")

    print("\nСтолбцы:")
    print(", ".join(columns))


if __name__ == "__main__":
    main()
