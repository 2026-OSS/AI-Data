import os
import unittest
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from server.model_predictor import (
    DEFAULT_HAND_LANDMARKER_MODEL_PATH,
    DEFAULT_YOLO_DATA_YAML,
    DEFAULT_YOLO_MODEL_PATH,
    ModelPredictor,
    PROJECT_ROOT,
    _import_dependency,
    _load_keras_model,
    _load_page_classes,
    _optional_float_from_env,
    _optional_int_from_env,
    _page_classes_from_env,
    _path_from_env,
    select_object_at_finger,
)
from server.predictor import PredictionUnavailable
from server.yolo_response import convert_ultralytics_result


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeBoxes:
    xyxy = FakeTensor([[10, 20, 100, 200]])
    conf = FakeTensor([0.88])
    cls = FakeTensor([1])


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "book_monkey", 1: "text"}


class FakeYoloModel:
    def __init__(self):
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [FakeResult()]


class FakeKerasModels:
    def __init__(self):
        self.loaded_paths = []

    def load_model(self, path):
        self.loaded_paths.append(path)
        with zipfile.ZipFile(path, "r") as model_zip:
            config = json.loads(model_zip.read("config.json"))
        dense_config = config["config"]["layers"][0]["config"]
        if "quantization_config" in dense_config:
            raise TypeError(
                "Unrecognized keyword arguments passed to Dense: "
                "{'quantization_config': None}"
            )
        return {"loaded": Path(path).name}


class FakeTensorflow:
    def __init__(self):
        self.keras = type("Keras", (), {"models": FakeKerasModels()})()


class FakeCv2:
    COLOR_BGR2RGB = 1

    def cvtColor(self, image, code):
        return image[:, :, ::-1]

    def resize(self, image, size):
        width, height = size
        return np.full((height, width, 3), 255, dtype=np.uint8)


class FakePageModel:
    input_shape = (None, 224, 224, 3)


class FakePredictingPageModel:
    def __init__(self, predictions):
        self.predictions = list(predictions)

    def predict(self, image, verbose=0):
        return np.asarray([self.predictions.pop(0)], dtype=np.float32)


class ModelPredictorTest(unittest.TestCase):
    def test_uses_yolo11_v15_as_default(self):
        self.assertEqual(
            DEFAULT_YOLO_MODEL_PATH.relative_to(PROJECT_ROOT),
            Path("artifacts/yolo11-v15/weights/best.pt"),
        )
        self.assertEqual(
            DEFAULT_YOLO_DATA_YAML.relative_to(PROJECT_ROOT),
            Path("artifacts/yolo11-v15/configs/data.yaml"),
        )
        self.assertEqual(
            DEFAULT_HAND_LANDMARKER_MODEL_PATH.relative_to(PROJECT_ROOT),
            Path("artifacts/hand-landmarker/hand_landmarker.task"),
        )
        self.assertTrue(DEFAULT_YOLO_MODEL_PATH.is_absolute())

    def test_resolves_relative_env_paths_from_project_root(self):
        with patch.dict("os.environ", {"YOLO_MODEL_PATH": "artifacts/custom/best.pt"}):
            path = _path_from_env("YOLO_MODEL_PATH", DEFAULT_YOLO_MODEL_PATH)

        self.assertEqual(path, PROJECT_ROOT / "artifacts/custom/best.pt")

    def test_reads_optional_numeric_env_values(self):
        with patch.dict(
            "os.environ",
            {"YOLO_IOU": "0.6", "YOLO_MAX_DET": "8", "EMPTY_VALUE": ""},
        ):
            self.assertEqual(_optional_float_from_env("YOLO_IOU"), 0.6)
            self.assertEqual(_optional_int_from_env("YOLO_MAX_DET"), 8)
            self.assertIsNone(_optional_float_from_env("EMPTY_VALUE"))

    def test_uses_fixed_page_label(self):
        predictor = object.__new__(ModelPredictor)
        predictor.fixed_page_label = "page2"
        predictor.fixed_page_confidence = 0.7

        page = predictor._predict_page(None)

        self.assertEqual(page, {"label": "page2", "confidence": 0.7})

    def test_returns_none_when_hands_are_disabled(self):
        predictor = object.__new__(ModelPredictor)
        predictor.disable_hands = True

        finger = predictor._predict_finger(None)

        self.assertIsNone(finger)

    def test_predict_preserves_objects_for_backend_matching(self):
        predictor = object.__new__(ModelPredictor)
        predictor._decode_image = lambda image_bytes: "image"
        predictor._predict_page = lambda image: {"label": "page2", "confidence": 0.9}
        predictor._predict_objects = lambda image: [
            {
                "label": "book_monkey",
                "confidence": 0.8,
                "bbox": [10, 20, 100, 120],
            }
        ]
        predictor._predict_finger = lambda image: {"x": 200, "y": 200}

        response = predictor.predict(b"image-bytes")

        self.assertEqual(
            response,
            {
                "page": {"label": "page2", "confidence": 0.9},
                "objects": [
                    {
                        "label": "book_monkey",
                        "confidence": 0.8,
                        "bbox": [10.0, 20.0, 100.0, 120.0],
                    }
                ],
                "finger": {"x": 200.0, "y": 200.0},
            },
        )

    def test_converts_ultralytics_result(self):
        objects = convert_ultralytics_result(FakeResult())

        self.assertEqual(
            objects,
            [
                {
                    "label": "text",
                    "confidence": 0.88,
                    "bbox": [10.0, 20.0, 100.0, 200.0],
                }
            ],
        )

    def test_predict_objects_passes_tuned_yolo_options(self):
        predictor = object.__new__(ModelPredictor)
        predictor.yolo_model = FakeYoloModel()
        predictor.yolo_class_names = ["book_monkey", "text"]
        predictor.yolo_conf = 0.15
        predictor.yolo_imgsz = 960
        predictor.yolo_iou = 0.6
        predictor.yolo_max_det = 8
        predictor.yolo_device = "mps"

        objects = predictor._predict_objects("image")

        self.assertEqual(objects[0]["label"], "text")
        self.assertEqual(
            predictor.yolo_model.calls,
            [
                (
                    "image",
                    {
                        "conf": 0.15,
                        "imgsz": 960,
                        "iou": 0.6,
                        "max_det": 8,
                        "device": "mps",
                        "verbose": False,
                    },
                )
            ],
        )

    def test_prepare_page_input_keeps_pixel_scale_for_model_rescaling_layer(self):
        predictor = object.__new__(ModelPredictor)
        predictor.page_model = FakePageModel()
        predictor._cv2 = FakeCv2()
        predictor._np = np

        batch = predictor._prepare_page_input(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(float(batch.max()), 255.0)

    def test_page_prediction_includes_top_k_margin_and_reliability(self):
        predictor = object.__new__(ModelPredictor)
        predictor.fixed_page_label = None
        predictor.page_model = FakePredictingPageModel([[0.02, 0.9, 0.05, 0.03]])
        predictor.page_classes = ["none", "page1", "page2", "page3"]
        predictor.page_confidence_threshold = 0.75
        predictor.page_margin_threshold = 0.15
        predictor.page_smoothing_alpha = 0.35
        predictor.page_stable_frames = 1
        predictor.page_none_confidence_threshold = None
        predictor._page_candidate_label = None
        predictor._page_candidate_count = 0
        predictor._accepted_page_label = None
        predictor._page_probability_ema = None
        predictor._prepare_page_input = lambda image: "input"

        page = predictor._predict_page("image")

        self.assertEqual(page["label"], "page1")
        self.assertTrue(page["reliable"])
        self.assertEqual(page["top_k"][0]["label"], "page1")
        self.assertAlmostEqual(page["margin"], 0.85, places=6)
        self.assertEqual(page["raw"]["label"], "page1")

    def test_page_prediction_lowers_confidence_when_raw_frame_disagrees(self):
        predictor = object.__new__(ModelPredictor)
        predictor.fixed_page_label = None
        predictor.page_model = FakePredictingPageModel(
            [
                [0.02, 0.9, 0.05, 0.03],
                [0.02, 0.05, 0.9, 0.03],
            ]
        )
        predictor.page_classes = ["none", "page1", "page2", "page3"]
        predictor.page_confidence_threshold = 0.75
        predictor.page_margin_threshold = 0.15
        predictor.page_smoothing_alpha = 0.2
        predictor.page_stable_frames = 1
        predictor.page_none_confidence_threshold = None
        predictor._page_candidate_label = None
        predictor._page_candidate_count = 0
        predictor._accepted_page_label = None
        predictor._page_probability_ema = None
        predictor._prepare_page_input = lambda image: "input"

        predictor._predict_page("image")
        page = predictor._predict_page("image")

        self.assertEqual(page["label"], "page1")
        self.assertEqual(page["raw"]["label"], "page2")
        self.assertFalse(page["reliable"])
        self.assertLess(page["confidence"], predictor.page_confidence_threshold)

    def test_page_prediction_requires_stable_consecutive_frames(self):
        predictor = object.__new__(ModelPredictor)
        predictor.fixed_page_label = None
        predictor.page_model = FakePredictingPageModel(
            [
                [0.02, 0.9, 0.05, 0.03],
                [0.02, 0.9, 0.05, 0.03],
            ]
        )
        predictor.page_classes = ["none", "page1", "page2", "page3"]
        predictor.page_confidence_threshold = 0.75
        predictor.page_margin_threshold = 0.15
        predictor.page_smoothing_alpha = 1.0
        predictor.page_stable_frames = 2
        predictor.page_none_confidence_threshold = None
        predictor._page_candidate_label = None
        predictor._page_candidate_count = 0
        predictor._accepted_page_label = None
        predictor._page_probability_ema = None
        predictor._prepare_page_input = lambda image: "input"

        first_page = predictor._predict_page("image")
        second_page = predictor._predict_page("image")

        self.assertFalse(first_page["reliable"])
        self.assertLess(first_page["confidence"], predictor.page_confidence_threshold)
        self.assertTrue(second_page["reliable"])
        self.assertEqual(second_page["label"], "page1")
        self.assertEqual(second_page["debug"]["page_candidate_count"], 2)

    def test_page_prediction_keeps_last_accepted_label_during_unstable_change(self):
        predictor = object.__new__(ModelPredictor)
        predictor.fixed_page_label = None
        predictor.page_model = FakePredictingPageModel(
            [
                [0.02, 0.9, 0.05, 0.03],
                [0.02, 0.9, 0.05, 0.03],
                [0.02, 0.05, 0.9, 0.03],
            ]
        )
        predictor.page_classes = ["none", "page1", "page2", "page3"]
        predictor.page_confidence_threshold = 0.75
        predictor.page_margin_threshold = 0.15
        predictor.page_smoothing_alpha = 1.0
        predictor.page_stable_frames = 2
        predictor.page_none_confidence_threshold = None
        predictor._page_candidate_label = None
        predictor._page_candidate_count = 0
        predictor._accepted_page_label = None
        predictor._page_probability_ema = None
        predictor._prepare_page_input = lambda image: "input"

        predictor._predict_page("image")
        predictor._predict_page("image")
        page = predictor._predict_page("image")

        self.assertEqual(page["smoothed"]["label"], "page2")
        self.assertEqual(page["label"], "page1")
        self.assertFalse(page["reliable"])

    def test_page_none_prediction_can_use_stricter_threshold(self):
        predictor = object.__new__(ModelPredictor)
        predictor.page_confidence_threshold = 0.75
        predictor.page_margin_threshold = 0.15
        predictor.page_none_confidence_threshold = 0.95

        reliable = predictor._is_reliable_page_prediction(
            {"label": "none", "confidence": 0.9, "margin": 0.8},
            {"label": "none", "confidence": 0.9, "margin": 0.8},
        )

        self.assertFalse(reliable)

    def test_create_from_env_uses_default_none_threshold(self):
        with patch.dict(os.environ, {}, clear=False):
            with patch.object(ModelPredictor, "__init__", return_value=None) as mock_init:
                ModelPredictor.from_env()

        self.assertEqual(mock_init.call_args.kwargs["page_none_confidence_threshold"], 0.9)

    def test_loads_page_classes_from_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "page_classifier_class_names.json"
            labels_path.write_text(
                json.dumps(["none", "page1", "page2", "page3"]),
                encoding="utf-8",
            )

            self.assertEqual(
                _load_page_classes(labels_path),
                ["none", "page1", "page2", "page3"],
            )

    def test_reads_page_classes_file_from_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_path = Path(tmpdir) / "labels.txt"
            labels_path.write_text("none\npage1\npage2\npage3\n", encoding="utf-8")

            with patch.dict("os.environ", {"PAGE_CLASSES_FILE": str(labels_path)}):
                class_names = _page_classes_from_env()

        self.assertEqual(class_names, ["none", "page1", "page2", "page3"])

    def test_selects_object_containing_finger(self):
        objects = [
            {"label": "book_monkey", "confidence": 0.7, "bbox": [10, 20, 100, 120]},
            {"label": "text", "confidence": 0.9, "bbox": [130, 20, 220, 120]},
        ]

        selected = select_object_at_finger(objects, {"x": 50, "y": 60})

        self.assertEqual(selected["label"], "book_monkey")

    def test_selects_nearby_object_with_margin(self):
        objects = [
            {"label": "book_monkey", "confidence": 0.7, "bbox": [10, 20, 100, 120]},
            {"label": "text", "confidence": 0.9, "bbox": [130, 20, 220, 120]},
        ]

        selected = select_object_at_finger(objects, {"x": 112, "y": 60}, margin=20)

        self.assertEqual(selected["label"], "book_monkey")

    def test_returns_none_when_finger_is_not_near_object(self):
        objects = [
            {"label": "book_monkey", "confidence": 0.7, "bbox": [10, 20, 100, 120]},
        ]

        selected = select_object_at_finger(objects, {"x": 160, "y": 60}, margin=20)

        self.assertIsNone(selected)

    def test_loads_keras_model_after_removing_quantization_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "page_model.keras"
            with zipfile.ZipFile(model_path, "w") as model_zip:
                model_zip.writestr("metadata.json", "{}")
                model_zip.writestr(
                    "config.json",
                    json.dumps(
                        {
                            "config": {
                                "layers": [
                                    {
                                        "config": {
                                            "units": 4,
                                            "quantization_config": None,
                                        }
                                    }
                                ]
                            }
                        }
                    ),
                )
                model_zip.writestr("model.weights.h5", b"weights")

            tensorflow = FakeTensorflow()

            model = _load_keras_model(tensorflow, model_path)

            self.assertEqual(model, {"loaded": "page_model.keras"})
            self.assertEqual(len(tensorflow.keras.models.loaded_paths), 2)

    def test_import_dependency_returns_requested_submodule(self):
        module = _import_dependency("xml.etree.ElementTree", "elementtree")

        self.assertEqual(module.__name__, "xml.etree.ElementTree")

    def test_import_dependency_raises_prediction_unavailable(self):
        with self.assertRaises(PredictionUnavailable):
            _import_dependency("missing_package_for_ai_data_tests", "missing")


if __name__ == "__main__":
    unittest.main()
