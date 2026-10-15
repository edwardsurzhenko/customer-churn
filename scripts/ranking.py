"""Ранжирование клиентов и оценка списка фиксированного размера."""

from numbers import Integral

import numpy as np
import pandas as pd


def rank_customers(customer_ids, probabilities):
    """Сортировать без целевой переменной: риск убывает, ID разрешает равенства."""
    ids = np.asarray(customer_ids, dtype=object)
    scores = np.asarray(probabilities, dtype=float)
    if ids.ndim != 1 or scores.ndim != 1 or len(ids) == 0 or len(ids) != len(scores):
        raise ValueError("Нужны непустые одномерные ID и вероятности одинаковой длины.")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("customerID должен быть непустой строкой.")
    if len(set(ids)) != len(ids):
        raise ValueError("customerID должен быть уникальным.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Вероятности должны быть конечными числами от 0 до 1.")
    ranked = pd.DataFrame({"customerID": ids, "risk_probability": scores})
    ranked = ranked.sort_values(["risk_probability", "customerID"],
                                ascending=[False, True], ignore_index=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def validate_k_values(k_values, size):
    values = list(k_values)
    if not values or any(isinstance(k, bool) or not isinstance(k, Integral)
                         or not 1 <= k <= size for k in values):
        raise ValueError(f"Каждое K должно быть целым числом от 1 до {size}.")
    if len(set(values)) != len(values):
        raise ValueError("Значения K не должны повторяться.")
    return [int(k) for k in values]


def evaluate_top_k(customer_ids, probabilities, target, k_values):
    """Оценить уже определённый порядок; target используется только в метриках."""
    ranked = rank_customers(customer_ids, probabilities)
    labels = np.asarray(target)
    if labels.ndim != 1 or len(labels) != len(ranked) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Целевая переменная должна содержать 0/1 для каждого клиента.")
    k_values = validate_k_values(k_values, len(ranked))
    by_id = pd.Series(labels.astype(int), index=np.asarray(customer_ids))
    ordered = by_id.loc[ranked["customerID"]].to_numpy()
    scores = ranked["risk_probability"].to_numpy()
    cumulative = ordered.cumsum()
    positives = int(ordered.sum())
    prevalence = positives / len(ranked)
    results = []
    for k in k_values:
        found = int(cumulative[k - 1])
        cutoff = scores[k - 1]
        above = scores > cutoff
        tied = scores == cutoff
        selected_from_tie = k - int(above.sum())
        tied_positive = int(ordered[tied].sum())
        tied_negative = int(tied.sum()) - tied_positive
        above_positive = int(ordered[above].sum())
        # Границы описывают чувствительность метрик к равным оценкам, не меняют отбор.
        minimum = above_positive + max(0, selected_from_tie - tied_negative)
        maximum = above_positive + min(selected_from_tie, tied_positive)
        results.append({
            "k": k, "population_fraction": k / len(ranked),
            "true_positives": found, "false_positives": k - found,
            "precision_at_k": found / k,
            "recall_at_k": found / positives if positives else None,
            "lift_at_k": (found / k) / prevalence if positives else None,
            "cutoff_probability": float(cutoff),
            "cutoff_tie_size": int(tied.sum()),
            "selected_from_cutoff_tie": selected_from_tie,
            "true_positive_bounds_within_cutoff_tie": [minimum, maximum],
        })
    return results


def random_selection_reference(population_size, positives, k_values):
    """Математическое ожидание равномерного выбора K клиентов без возвращения."""
    if (isinstance(population_size, bool) or not isinstance(population_size, Integral)
            or population_size <= 0 or isinstance(positives, bool)
            or not isinstance(positives, Integral) or not 0 <= positives <= population_size):
        raise ValueError("Нужны размер выборки > 0 и число положительных от 0 до размера.")
    prevalence = positives / population_size
    return [{
        "k": k, "expected_true_positives": k * prevalence,
        "expected_precision_at_k": prevalence,
        "expected_recall_at_k": k / population_size if positives else None,
        "expected_lift_at_k": 1.0 if positives else None,
    } for k in validate_k_values(k_values, population_size)]
