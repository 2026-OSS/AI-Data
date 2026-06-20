import unittest
import json
import tempfile
import zipfile
from pathlib import Path

from server.model_predictor import (
    DEFAULT_HAND_LANDMARKER_MODEL_PATH,
    DEFAULT_YOLO_DATA_YAML,
    DEFAULT_YOLO_MODEL_PATH,
    ModelPredictor,
    _import_dependency,
    _load_keras_model,
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


class ModelPredictorTest(unittest.TestCase):
    def test_uses_yolo11_v15_as_default(self):
        self.assertEqual(
            str(DEFAULT_YOLO_MODEL_PATH),
            "artifacts/yolo11-v15/weights/best.pt",
        )
        self.assertEqual(
            str(DEFAULT_YOLO_DATA_YAML),
            "artifacts/yolo11-v15/configs/data.yaml",
        )
        self.assertEqual(
            str(DEFAULT_HAND_LANDMARKER_MODEL_PATH),
            "artifacts/hand-landmarker/hand_landmarker.task",
        )

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
