from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence
from importlib import import_module

from server.backend_response import build_backend_response
from server.hand_response import extract_finger_from_results
from server.page_response import DEFAULT_PAGE_CLASSES, convert_page_prediction
from server.predictor import PredictionUnavailable
from server.yolo_response import (
    convert_ultralytics_result,
    load_class_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def _path_from_env(name: str, default_path: Path) -> Path:
    configured_path = os.getenv(name)
    if configured_path is None:
        return default_path

    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_float_from_env(name: str) -> float | None:
    value = os.getenv(name)
    return None if value in (None, "") else float(value)


def _optional_int_from_env(name: str) -> int | None:
    value = os.getenv(name)
    return None if value in (None, "") else int(value)


DEFAULT_YOLO_MODEL_PATH = _project_path("artifacts/yolo11-v15/weights/best.pt")
DEFAULT_YOLO_DATA_YAML = _project_path("artifacts/yolo11-v15/configs/data.yaml")
DEFAULT_PAGE_MODEL_PATH = _project_path(
    "artifacts/page-classifier-mobilenetv2/page_classifier_mobilenetv2.keras"
)
DEFAULT_HAND_LANDMARKER_MODEL_PATH = _project_path(
    "artifacts/hand-landmarker/hand_landmarker.task"
)


class ModelPredictor:
    def __init__(
        self,
        yolo_model_path: str | Path = DEFAULT_YOLO_MODEL_PATH,
        page_model_path: str | Path = DEFAULT_PAGE_MODEL_PATH,
        yolo_data_yaml: str | Path = DEFAULT_YOLO_DATA_YAML,
        hand_landmarker_model_path: str | Path = DEFAULT_HAND_LANDMARKER_MODEL_PATH,
        page_classes: list[str] | None = None,
        fixed_page_label: str | None = None,
        fixed_page_confidence: float = 1.0,
        disable_hands: bool = False,
        yolo_conf: float = 0.15,
        yolo_imgsz: int = 960,
        yolo_iou: float | None = None,
        yolo_max_det: int | None = None,
        yolo_device: str | None = None,
        hand_min_detection_confidence: float = 0.4,
        hand_min_presence_confidence: float = 0.4,
        hand_min_tracking_confidence: float = 0.4,
        finger_smoothing_alpha: float = 0.55,
        finger_missing_grace_frames: int = 2,
        page_confidence_threshold: float = 0.75,
        page_margin_threshold: float = 0.15,
        page_smoothing_alpha: float = 0.35,
        page_stable_frames: int = 2,
        page_none_confidence_threshold: float | None = None,
    ) -> None:
        self.yolo_model_path = Path(yolo_model_path)
        self.page_model_path = Path(page_model_path)
        self.hand_landmarker_model_path = Path(hand_landmarker_model_path)
        self.yolo_class_names = load_class_names(yolo_data_yaml)
        self.page_classes = page_classes or DEFAULT_PAGE_CLASSES
        self.fixed_page_label = fixed_page_label
        self.fixed_page_confidence = fixed_page_confidence
        self.disable_hands = disable_hands
        self.yolo_conf = yolo_conf
        self.yolo_imgsz = yolo_imgsz
        self.yolo_iou = yolo_iou
        self.yolo_max_det = yolo_max_det
        self.yolo_device = yolo_device or None
        self.hand_min_detection_confidence = hand_min_detection_confidence
        self.hand_min_presence_confidence = hand_min_presence_confidence
        self.hand_min_tracking_confidence = hand_min_tracking_confidence
        self.finger_smoothing_alpha = finger_smoothing_alpha
        self.finger_missing_grace_frames = max(int(finger_missing_grace_frames), 0)
        self.page_confidence_threshold = page_confidence_threshold
        self.page_margin_threshold = page_margin_threshold
        self.page_smoothing_alpha = page_smoothing_alpha
        self.page_stable_frames = max(int(page_stable_frames), 1)
        self.page_none_confidence_threshold = page_none_confidence_threshold
        self._page_probability_ema: list[float] | None = None
        self._page_candidate_label: str | None = None
        self._page_candidate_count = 0
        self._accepted_page_label: str | None = None
        self._finger_ema: dict[str, float] | None = None
        self._finger_missing_count = 0

        self._cv2 = _import_dependency("cv2", "opencv-python")
        self._np = _import_dependency("numpy", "numpy")
        ultralytics = _import_dependency("ultralytics", "ultralytics")
        if self.disable_hands:
            mediapipe = None
        else:
            mediapipe = _import_dependency("mediapipe", "mediapipe")

        if not self.yolo_model_path.exists():
            raise PredictionUnavailable(f"YOLO model not found: {self.yolo_model_path}")
        if self.fixed_page_label is None and not self.page_model_path.exists():
            raise PredictionUnavailable(f"page model not found: {self.page_model_path}")
        if not self.disable_hands and not self.hand_landmarker_model_path.exists():
            raise PredictionUnavailable(
                f"hand landmarker model not found: {self.hand_landmarker_model_path}"
            )

        self.yolo_model = ultralytics.YOLO(str(self.yolo_model_path))
        if self.fixed_page_label is None:
            tensorflow = _import_dependency("tensorflow", "tensorflow")
            self.page_model = _load_keras_model(tensorflow, self.page_model_path)
        else:
            self.page_model = None
        if self.disable_hands:
            self.hands = None
        else:
            base_options = mediapipe.tasks.BaseOptions(
                model_asset_path=str(self.hand_landmarker_model_path),
                delegate=mediapipe.tasks.BaseOptions.Delegate.CPU,
            )
            options = mediapipe.tasks.vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mediapipe.tasks.vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=self.hand_min_detection_confidence,
                min_hand_presence_confidence=self.hand_min_presence_confidence,
                min_tracking_confidence=self.hand_min_tracking_confidence,
            )
            self._mediapipe = mediapipe
            self.hands = mediapipe.tasks.vision.HandLandmarker.create_from_options(
                options
            )

    @classmethod
    def from_env(cls) -> "ModelPredictor":
        return cls(
            yolo_model_path=_path_from_env("YOLO_MODEL_PATH", DEFAULT_YOLO_MODEL_PATH),
            page_model_path=_path_from_env("PAGE_MODEL_PATH", DEFAULT_PAGE_MODEL_PATH),
            yolo_data_yaml=_path_from_env("YOLO_DATA_YAML", DEFAULT_YOLO_DATA_YAML),
            hand_landmarker_model_path=_path_from_env(
                "HAND_LANDMARKER_MODEL_PATH", DEFAULT_HAND_LANDMARKER_MODEL_PATH
            ),
            page_classes=_page_classes_from_env(),
            fixed_page_label=os.getenv("PAGE_LABEL"),
            fixed_page_confidence=float(os.getenv("PAGE_CONFIDENCE", "1.0")),
            disable_hands=os.getenv("DISABLE_HANDS", "false").lower() == "true",
            yolo_conf=float(os.getenv("YOLO_CONF", "0.15")),
            yolo_imgsz=int(os.getenv("YOLO_IMGSZ", "960")),
            yolo_iou=_optional_float_from_env("YOLO_IOU"),
            yolo_max_det=_optional_int_from_env("YOLO_MAX_DET"),
            yolo_device=os.getenv("YOLO_DEVICE") or None,
            hand_min_detection_confidence=float(
                os.getenv("HAND_MIN_DETECTION_CONFIDENCE", "0.4")
            ),
            hand_min_presence_confidence=float(
                os.getenv("HAND_MIN_PRESENCE_CONFIDENCE", "0.4")
            ),
            hand_min_tracking_confidence=float(
                os.getenv("HAND_MIN_TRACKING_CONFIDENCE", "0.4")
            ),
            finger_smoothing_alpha=float(os.getenv("FINGER_SMOOTHING_ALPHA", "0.55")),
            finger_missing_grace_frames=int(
                os.getenv("FINGER_MISSING_GRACE_FRAMES", "2")
            ),
            page_confidence_threshold=float(
                os.getenv("PAGE_CONFIDENCE_THRESHOLD", "0.75")
            ),
            page_margin_threshold=float(os.getenv("PAGE_MARGIN_THRESHOLD", "0.15")),
            page_smoothing_alpha=float(os.getenv("PAGE_SMOOTHING_ALPHA", "0.35")),
            page_stable_frames=int(os.getenv("PAGE_STABLE_FRAMES", "2")),
            page_none_confidence_threshold=float(
                os.getenv("PAGE_NONE_CONFIDENCE_THRESHOLD", "0.90")
            ),
        )

    def predict(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        image = self._decode_image(image_bytes)
        page = self._predict_page(image)
        objects = self._predict_objects(image)
        finger = self._predict_finger(image)

        return build_backend_response(page=page, objects=objects, finger=finger)

    def _decode_image(self, image_bytes: bytes) -> Any:
        buffer = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
        image = self._cv2.imdecode(buffer, self._cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("frame image cannot be decoded")
        return image

    def _predict_page(self, image: Any) -> dict[str, Any]:
        if self.fixed_page_label is not None:
            return {
                "label": self.fixed_page_label,
                "confidence": self.fixed_page_confidence,
            }

        input_image = self._prepare_page_input(image)
        raw_probabilities = [
            float(value) for value in self.page_model.predict(input_image, verbose=0)[0]
        ]
        raw_page = convert_page_prediction(raw_probabilities, self.page_classes)

        smoothed_probabilities = self._smooth_page_probabilities(raw_probabilities)
        smoothed_page = convert_page_prediction(smoothed_probabilities, self.page_classes)

        reliable = self._is_reliable_page_prediction(raw_page, smoothed_page)
        stable = self._update_page_stability(str(smoothed_page["label"]), reliable)
        confidence = float(smoothed_page["confidence"])
        margin = float(smoothed_page["margin"])
        label = str(smoothed_page["label"])
        if not stable:
            confidence = min(confidence, self.page_confidence_threshold - 0.001)
            if self._accepted_page_label is not None:
                label = self._accepted_page_label

        return {
            "label": label,
            "confidence": max(0.0, confidence),
            "margin": margin,
            "reliable": stable,
            "top_k": smoothed_page["top_k"],
            "raw": raw_page,
            "smoothed": smoothed_page,
            "debug": {
                "page_confidence_threshold": self.page_confidence_threshold,
                "page_none_confidence_threshold": self.page_none_confidence_threshold,
                "page_margin_threshold": self.page_margin_threshold,
                "page_smoothing_alpha": self.page_smoothing_alpha,
                "page_stable_frames": self.page_stable_frames,
                "page_candidate_label": self._page_candidate_label,
                "page_candidate_count": self._page_candidate_count,
                "accepted_page_label": self._accepted_page_label,
            },
        }

    def _predict_objects(self, image: Any) -> list[dict[str, Any]]:
        predict_kwargs: dict[str, Any] = {
            "conf": self.yolo_conf,
            "imgsz": self.yolo_imgsz,
            "verbose": False,
        }
        if self.yolo_iou is not None:
            predict_kwargs["iou"] = self.yolo_iou
        if self.yolo_max_det is not None:
            predict_kwargs["max_det"] = self.yolo_max_det
        if self.yolo_device is not None:
            predict_kwargs["device"] = self.yolo_device

        result = self.yolo_model.predict(image, **predict_kwargs)[0]
        return convert_ultralytics_result(result, self.yolo_class_names)

    def _predict_finger(self, image: Any) -> dict[str, float] | None:
        if self.disable_hands:
            return None

        height, width = image.shape[:2]
        rgb_image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mediapipe.Image(
            image_format=self._mediapipe.ImageFormat.SRGB,
            data=rgb_image,
        )
        results = self.hands.detect(mp_image)
        finger = extract_finger_from_results(results, width, height)
        return self._smooth_finger(finger)

    def _smooth_finger(self, finger: dict[str, float] | None) -> dict[str, float] | None:
        if finger is None:
            if (
                self._finger_ema is not None
                and self._finger_missing_count < self.finger_missing_grace_frames
            ):
                self._finger_missing_count += 1
                return dict(self._finger_ema)

            self._finger_ema = None
            self._finger_missing_count = 0
            return None

        alpha = min(max(self.finger_smoothing_alpha, 0.0), 1.0)
        if self._finger_ema is None:
            self._finger_ema = {"x": float(finger["x"]), "y": float(finger["y"])}
        else:
            self._finger_ema = {
                "x": alpha * float(finger["x"]) + (1.0 - alpha) * self._finger_ema["x"],
                "y": alpha * float(finger["y"]) + (1.0 - alpha) * self._finger_ema["y"],
            }

        self._finger_missing_count = 0
        return dict(self._finger_ema)

    def _prepare_page_input(self, image: Any) -> Any:
        input_shape = self.page_model.input_shape
        height = input_shape[1] or 224
        width = input_shape[2] or 224

        rgb_image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        resized = self._cv2.resize(rgb_image, (width, height))
        batch = resized.astype("float32")
        return self._np.expand_dims(batch, axis=0)

    def _smooth_page_probabilities(
        self,
        probabilities: Sequence[float],
    ) -> list[float]:
        alpha = min(max(self.page_smoothing_alpha, 0.0), 1.0)
        current = [float(value) for value in probabilities]
        if self._page_probability_ema is None or len(self._page_probability_ema) != len(
            current
        ):
            self._page_probability_ema = current
            return current

        self._page_probability_ema = [
            alpha * current_value + (1.0 - alpha) * previous_value
            for current_value, previous_value in zip(current, self._page_probability_ema)
        ]
        return self._page_probability_ema

    def _is_reliable_page_prediction(
        self,
        raw_page: dict[str, Any],
        smoothed_page: dict[str, Any],
    ) -> bool:
        label = str(smoothed_page["label"])
        confidence_threshold = self.page_confidence_threshold
        if label == "none" and self.page_none_confidence_threshold is not None:
            confidence_threshold = self.page_none_confidence_threshold

        return (
            raw_page["label"] == label
            and float(smoothed_page["confidence"]) >= confidence_threshold
            and float(smoothed_page["margin"]) >= self.page_margin_threshold
        )

    def _update_page_stability(self, label: str, reliable: bool) -> bool:
        if not reliable:
            self._page_candidate_label = None
            self._page_candidate_count = 0
            return False

        if self._page_candidate_label == label:
            self._page_candidate_count += 1
        else:
            self._page_candidate_label = label
            self._page_candidate_count = 1

        if self._page_candidate_count >= self.page_stable_frames:
            self._accepted_page_label = label
            return True

        return False


def create_default_predictor() -> ModelPredictor:
    return ModelPredictor.from_env()


def _page_classes_from_env() -> list[str]:
    labels_file = os.getenv("PAGE_CLASSES_FILE") or os.getenv("PAGE_LABELS_FILE")
    if labels_file:
        path = Path(labels_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return _load_page_classes(path)

    return [
        name.strip()
        for name in os.getenv("PAGE_CLASSES", ",".join(DEFAULT_PAGE_CLASSES)).split(",")
        if name.strip()
    ]


def _load_page_classes(path: Path) -> list[str]:
    if not path.exists():
        raise PredictionUnavailable(f"page class names file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            names = data
        elif isinstance(data, dict) and isinstance(data.get("class_names"), list):
            names = data["class_names"]
        else:
            raise PredictionUnavailable(
                f"unsupported page class names JSON format: {path}"
            )
    else:
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return [str(name).strip() for name in names if str(name).strip()]


def select_object_at_finger(
    objects: list[dict[str, Any]],
    finger: dict[str, Any],
    margin: float = 0.0,
) -> dict[str, Any] | None:
    matches = [
        detected
        for detected in objects
        if _point_distance_to_bbox(finger, detected["bbox"]) <= margin
    ]
    if not matches:
        return None

    return min(
        matches,
        key=lambda detected: (
            _point_distance_to_bbox(finger, detected["bbox"]),
            -float(detected["confidence"]),
        ),
    )


def _point_distance_to_bbox(finger: dict[str, Any], bbox: list[float]) -> float:
    x = float(finger["x"])
    y = float(finger["y"])
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 0.0

    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


def _load_keras_model(tensorflow: Any, model_path: Path) -> Any:
    try:
        return tensorflow.keras.models.load_model(str(model_path))
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        with tempfile.TemporaryDirectory() as tmpdir:
            patched_path = Path(tmpdir) / model_path.name
            _copy_keras_model_without_quantization_config(model_path, patched_path)
            return tensorflow.keras.models.load_model(str(patched_path))


def _copy_keras_model_without_quantization_config(
    source_path: Path,
    target_path: Path,
) -> None:
    with zipfile.ZipFile(source_path, "r") as source_zip:
        config = json.loads(source_zip.read("config.json"))
        _remove_key(config, "quantization_config")

        with zipfile.ZipFile(target_path, "w") as target_zip:
            for info in source_zip.infolist():
                if info.filename == "config.json":
                    target_zip.writestr(
                        info,
                        json.dumps(config, ensure_ascii=False).encode("utf-8"),
                    )
                else:
                    target_zip.writestr(info, source_zip.read(info.filename))


def _remove_key(value: Any, key: str) -> None:
    if isinstance(value, dict):
        value.pop(key, None)
        for child in value.values():
            _remove_key(child, key)
    elif isinstance(value, list):
        for child in value:
            _remove_key(child, key)


def _import_dependency(module_name: str, package_name: str) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise PredictionUnavailable(
            f"{package_name} is required for prediction"
        ) from exc
