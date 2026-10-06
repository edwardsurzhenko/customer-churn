"""Скачать фиксированную версию учебного датасета IBM."""

import csv
import io
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
SOURCE_COMMIT = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{SOURCE_COMMIT}/data/Telco-Customer-Churn.csv"
)


def main():
    if DESTINATION.exists():
        print(f"Файл уже существует: {DESTINATION}")
        return

    try:
        with urlopen(SOURCE_URL, timeout=30) as response:
            data = response.read()
    except URLError as error:
        raise SystemExit(f"Не удалось скачать данные. Проверьте интернет: {error}")

    # Проверяем содержимое перед сохранением, чтобы не записать страницу ошибки.
    table = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    required_columns = {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}
    if not required_columns.issubset(table.fieldnames or []):
        raise SystemExit("Полученный файл не содержит ожидаемые столбцы датасета.")
    if not any(table):
        raise SystemExit("Полученный CSV не содержит строк с данными.")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(data)
    print(f"Данные сохранены: {DESTINATION}")


if __name__ == "__main__":
    main()
