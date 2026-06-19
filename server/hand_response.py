from __future__ import annotations

from typing import Any


INDEX_FINGER_TIP = 8


def convert_normalized_point(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    _validate_image_size(image_width, image_height)

    return {
        "x": float(x) * image_width,
        "y": float(y) * image_height,
    }


def extract_index_finger_tip(
    hand_landmarks: Any,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    _validate_image_size(image_width, image_height)

    landmarks = getattr(hand_landmarks, "landmark", hand_landmarks)
    if len(landmarks) <= INDEX_FINGER_TIP:
        raise ValueError("index finger tip landmark not found")

    tip = landmarks[INDEX_FINGER_TIP]
    return convert_normalized_point(tip.x, tip.y, image_width, image_height)


def extract_finger_from_results(
    results: Any,
    image_width: int,
    image_height: int,
) -> dict[str, float] | None:
    task_hands = getattr(results, "hand_landmarks", None)
    if task_hands:
        return extract_index_finger_tip(task_hands[0], image_width, image_height)

    hands = getattr(results, "multi_hand_landmarks", None)
    if not hands:
        return None

    return extract_index_finger_tip(hands[0], image_width, image_height)


def _validate_image_size(image_width: int, image_height: int) -> None:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive")
