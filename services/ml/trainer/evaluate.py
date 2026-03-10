from __future__ import annotations

from importlib import import_module
from typing import Any


def binary_classification_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, Any]:
    if not labels or not probabilities:
        return {
            "auroc": 0.0,
            "auprc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "ece": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "threshold": threshold,
            "count": 0,
        }

    np = import_module("numpy")
    metrics_module = import_module("sklearn.metrics")
    average_precision_score = getattr(metrics_module, "average_precision_score")
    confusion_matrix = getattr(metrics_module, "confusion_matrix")
    f1_score = getattr(metrics_module, "f1_score")
    precision_score = getattr(metrics_module, "precision_score")
    recall_score = getattr(metrics_module, "recall_score")
    roc_auc_score = getattr(metrics_module, "roc_auc_score")

    labels_array = np.asarray(labels, dtype=np.int32)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    predictions = (probability_array >= threshold).astype(np.int32)

    if len(set(labels)) > 1:
        auroc = float(roc_auc_score(labels_array, probability_array))
        auprc = float(average_precision_score(labels_array, probability_array))
    else:
        auroc = 0.5
        auprc = float(labels_array.mean())

    precision = float(precision_score(labels_array, predictions, zero_division=0))
    recall = float(recall_score(labels_array, predictions, zero_division=0))
    f1 = float(f1_score(labels_array, predictions, zero_division=0))
    confusion = confusion_matrix(labels_array, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    false_positive_rate = float(fp / max(fp + tn, 1))
    false_negative_rate = float(fn / max(fn + tp, 1))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "ece": expected_calibration_error(labels_array, probability_array),
        "confusion_matrix": confusion.tolist(),
        "threshold": threshold,
        "count": int(len(labels_array)),
    }


def expected_calibration_error(labels: Any, probabilities: Any, bins: int = 10) -> float:
    if len(labels) == 0:
        return 0.0

    np = import_module("numpy")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    ece = 0.0

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        bin_probabilities = probabilities[mask]
        bin_labels = labels[mask]
        confidence = float(bin_probabilities.mean())
        accuracy = float(bin_labels.mean())
        ece += (len(bin_labels) / total) * abs(accuracy - confidence)

    return float(ece)


def choose_threshold_with_target_fpr(labels: list[int], probabilities: list[float], max_false_positive_rate: float) -> float:
    if not labels or not probabilities:
        return 0.5

    candidates = sorted({round(value, 6) for value in probabilities}, reverse=True)
    best_threshold = 0.5
    best_score = (-1.0, -1.0)

    for candidate in candidates:
        metrics = binary_classification_metrics(labels, probabilities, candidate)
        if metrics["false_positive_rate"] <= max_false_positive_rate:
            score = (metrics["recall"], metrics["precision"])
            if score > best_score:
                best_score = score
                best_threshold = candidate

    return best_threshold


def probability_band_thresholds(elevated_threshold: float) -> dict[str, float]:
    review_threshold = min(elevated_threshold, max(0.25, elevated_threshold - 0.15))
    return {
        "review_threshold": round(review_threshold, 4),
        "elevated_threshold": round(elevated_threshold, 4),
    }


def build_slice_reports(
    rows: list[dict[str, Any]],
    *,
    slice_keys: list[str],
    probability_key: str,
    threshold: float,
    min_examples: int,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for slice_key in slice_keys:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            value = str(row.get(slice_key, "unknown"))
            grouped.setdefault(value, []).append(row)

        for value, grouped_rows in grouped.items():
            if len(grouped_rows) < min_examples:
                continue
            labels = [int(row["label"]) for row in grouped_rows]
            probabilities = [float(row[probability_key]) for row in grouped_rows]
            metrics = binary_classification_metrics(labels, probabilities, threshold)
            reports.append(
                {
                    "slice_key": slice_key,
                    "label": value,
                    "count": len(grouped_rows),
                    **metrics,
                }
            )
    return reports
