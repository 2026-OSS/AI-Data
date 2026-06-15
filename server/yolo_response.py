from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_YAML = Path("artifacts/yolov5-v2/data.yaml")


def load_class_names(data_yaml_path: str | Path = DEFAULT_DATA_YAML) -> list[str]:
    path = Path(data_yaml_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    names: list[str] = []
    in_names = False

    for line in lines:
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            continue

        if in_names:
            if not stripped:
                continue
            if not stripped.startswith("-"):
                break
            names.append(stripped.removeprefix("-").strip().strip("'\""))

    if not names:
        raise ValueError(f"class names not found in {path}")

    return names


def convert_detection(
    detection: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    class_id = int(detection.get("class_id", detection.get("cls")))
    confidence = float(detection.get("confidence", detection.get("conf")))
    bbox = detection.get("bbox", detection.get("xyxy"))

    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must contain [x1, y1, x2, y2]")

    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x1 > x2 or y1 > y2:
        raise ValueError("bbox must use [x1, y1, x2, y2] order")

    return {
        "label": class_names[class_id],
        "confidence": confidence,
        "bbox": [x1, y1, x2, y2],
    }


def convert_detections(
    detections: Iterable[dict[str, Any]],
    class_names: list[str],
) -> list[dict[str, Any]]:
    return [convert_detection(detection, class_names) for detection in detections]


def convert_yolov5_result(
    result: Any,
    class_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    names = class_names or _resolve_result_names(result)
    rows = result.xyxy[0].tolist()

    detections = [
        {
            "xyxy": row[:4],
            "confidence": row[4],
            "class_id": row[5],
        }
        for row in rows
    ]

    return convert_detections(detections, names)


def _resolve_result_names(result: Any) -> list[str]:
    names = result.names
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)
