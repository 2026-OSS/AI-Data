from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from importlib import import_module

from server.backend_response import build_backend_response
from server.hand_response import extract_finger_from_results
from server.page_response import DEFAULT_PAGE_CLASSES, convert_page_prediction
from server.predictor import PredictionUnavailable
from server.yolo_response import (
    convert_ultralytics_result,
    load_class_names,
)


DEFAULT_YOLO_MODEL_PATH = Path("artifacts/yolo11-v15/weights/best.pt")
DEFAULT_YOLO_DATA_YAML = Path("artifacts/yolo11-v15/configs/data.yaml")
DEFAULT_PAGE_MODEL_PATH = Path(
    "artifacts/page-classifier-mobilenetv2/page_classifier_mobilenetv2.keras"
)
DEFAULT_HAND_LANDMARKER_MODEL_PATH = Path(
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
    ) -> None:
        self.yolo_model_path = Path(yolo_model_path)
        self.page_model_path = Path(page_model_path)
        self.hand_landmarker_model_path = Path(hand_landmarker_model_path)
        self.yolo_class_names = load_class_names(yolo_data_yaml)
        self.page_classes = page_classes or DEFAULT_PAGE_CLASSES
        self.fixed_page_label = fixed_page_label
        self.fixed_page_confidence = fixed_page_confidence
        self.disable_hands = disable_hands

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
            )
            self._mediapipe = mediapipe
            self.hands = mediapipe.tasks.vision.HandLandmarker.create_from_options(
                options
            )

    @classmethod
    def from_env(cls) -> "ModelPredictor":
        return cls(
            yolo_model_path=os.getenv("YOLO_MODEL_PATH", str(DEFAULT_YOLO_MODEL_PATH)),
            page_model_path=os.getenv("PAGE_MODEL_PATH", str(DEFAULT_PAGE_MODEL_PATH)),
            yolo_data_yaml=os.getenv("YOLO_DATA_YAML", str(DEFAULT_YOLO_DATA_YAML)),
            hand_landmarker_model_path=os.getenv(
                "HAND_LANDMARKER_MODEL_PATH",
                str(DEFAULT_HAND_LANDMARKER_MODEL_PATH),
            ),
            page_classes=os.getenv("PAGE_CLASSES", ",".join(DEFAULT_PAGE_CLASSES)).split(
                ","
            ),
            fixed_page_label=os.getenv("PAGE_LABEL"),
            fixed_page_confidence=float(os.getenv("PAGE_CONFIDENCE", "1.0")),
            disable_hands=os.getenv("DISABLE_HANDS", "false").lower() == "true",
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
        probabilities = self.page_model.predict(input_image, verbose=0)[0]
        return convert_page_prediction(probabilities, self.page_classes)

    def _predict_objects(self, image: Any) -> list[dict[str, Any]]:
        result = self.yolo_model(image, verbose=False)[0]
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
        return extract_finger_from_results(results, width, height)

    def _prepare_page_input(self, image: Any) -> Any:
        input_shape = self.page_model.input_shape
        height = input_shape[1] or 224
        width = input_shape[2] or 224

        rgb_image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        resized = self._cv2.resize(rgb_image, (width, height))
        normalized = resized.astype("float32") / 255.0
        return self._np.expand_dims(normalized, axis=0)


def create_default_predictor() -> ModelPredictor:
    return ModelPredictor.from_env()


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
