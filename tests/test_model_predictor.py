import unittest

from server.model_predictor import (
    DEFAULT_YOLO_DATA_YAML,
    DEFAULT_YOLO_MODEL_PATH,
    ModelPredictor,
)
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


class ModelPredictorTest(unittest.TestCase):
    def test_uses_yolo11_v12_as_default(self):
        self.assertEqual(str(DEFAULT_YOLO_MODEL_PATH), "artifacts/yolo11-v12/best.pt")
        self.assertEqual(str(DEFAULT_YOLO_DATA_YAML), "artifacts/yolo11-v12/data.yaml")

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


if __name__ == "__main__":
    unittest.main()
