from __future__ import annotations

from typing import Sequence


DEFAULT_PAGE_CLASSES = ["none", "page1", "page2", "page3"]


def convert_page_prediction(
    probabilities: Sequence[float],
    class_names: Sequence[str] = DEFAULT_PAGE_CLASSES,
) -> dict[str, float | str]:
    if len(probabilities) != len(class_names):
        raise ValueError("probabilities and class_names must have the same length")

    if not probabilities:
        raise ValueError("probabilities must not be empty")

    best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])

    return {
        "label": class_names[best_index],
        "confidence": float(probabilities[best_index]),
    }


def convert_page_label(
    label: str,
    confidence: float,
    class_names: Sequence[str] = DEFAULT_PAGE_CLASSES,
) -> dict[str, float | str]:
    if label not in class_names:
        raise ValueError(f"unknown page label: {label}")

    return {
        "label": label,
        "confidence": float(confidence),
    }
