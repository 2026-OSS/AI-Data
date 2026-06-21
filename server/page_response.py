from __future__ import annotations

from typing import Sequence


DEFAULT_PAGE_CLASSES = ["none", "page1", "page2", "page3"]


def convert_page_prediction(
    probabilities: Sequence[float],
    class_names: Sequence[str] = DEFAULT_PAGE_CLASSES,
) -> dict[str, float | str | bool | list[dict[str, float | str]]]:
    if len(probabilities) != len(class_names):
        raise ValueError("probabilities and class_names must have the same length")

    if len(probabilities) == 0:
        raise ValueError("probabilities must not be empty")

    ranked_indices = sorted(
        range(len(probabilities)),
        key=lambda index: float(probabilities[index]),
        reverse=True,
    )
    best_index = ranked_indices[0]
    best_confidence = float(probabilities[best_index])
    runner_up_confidence = (
        float(probabilities[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0
    )

    return {
        "label": class_names[best_index],
        "confidence": best_confidence,
        "margin": best_confidence - runner_up_confidence,
        "top_k": [
            {"label": class_names[index], "confidence": float(probabilities[index])}
            for index in ranked_indices
        ],
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
