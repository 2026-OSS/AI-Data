from __future__ import annotations

from typing import Any, Iterable


def build_backend_response(
    page: dict[str, Any],
    objects: Iterable[dict[str, Any]],
    finger: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "page": normalize_page(page),
        "objects": [normalize_object(detected) for detected in objects],
        "finger": normalize_finger(finger),
    }


def normalize_page(page: dict[str, Any]) -> dict[str, Any]:
    label = page.get("label", page.get("page"))
    confidence = page.get("confidence")

    if not label:
        raise ValueError("page label is required")

    normalized = {
        "label": str(label),
        "confidence": _normalize_confidence(confidence),
    }
    for key in (
        "margin",
        "reliable",
        "raw",
        "smoothed",
        "top_k",
        "debug",
    ):
        if key in page:
            normalized[key] = page[key]

    return normalized


def normalize_object(detected: dict[str, Any]) -> dict[str, Any]:
    label = detected.get("label", detected.get("class", detected.get("class_name")))
    confidence = detected.get("confidence")
    bbox = detected.get("bbox")

    if not label:
        raise ValueError("object label is required")
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")

    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x1 > x2 or y1 > y2:
        raise ValueError("bbox must use [x1, y1, x2, y2] order")

    return {
        "label": str(label),
        "confidence": _normalize_confidence(confidence),
        "bbox": [x1, y1, x2, y2],
    }


def normalize_finger(finger: dict[str, Any] | None) -> dict[str, float] | None:
    if finger is None:
        return None

    if "x" not in finger or "y" not in finger:
        raise ValueError("finger must contain x and y")

    return {
        "x": float(finger["x"]),
        "y": float(finger["y"]),
    }


def _normalize_confidence(confidence: Any) -> float:
    value = float(confidence)
    if value < 0 or value > 1:
        raise ValueError("confidence must be between 0 and 1")
    return value
