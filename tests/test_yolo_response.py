import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from server.yolo_response import convert_detections, load_class_names


class YoloResponseTest(unittest.TestCase):
    def test_load_class_names_from_data_yaml(self):
        with TemporaryDirectory() as temp_dir:
            data_yaml = Path(temp_dir) / "data.yaml"
            data_yaml.write_text(
                "names:\n- book_monkey\n- braille\nnc: 2\n",
                encoding="utf-8",
            )

            self.assertEqual(load_class_names(data_yaml), ["book_monkey", "braille"])

    def test_convert_detections_to_backend_objects(self):
        objects = convert_detections(
            [
                {
                    "class_id": 1,
                    "confidence": 0.95,
                    "bbox": [120, 85, 300, 410],
                }
            ],
            ["book_monkey", "braille"],
        )

        self.assertEqual(
            objects,
            [
                {
                    "label": "braille",
                    "confidence": 0.95,
                    "bbox": [120.0, 85.0, 300.0, 410.0],
                }
            ],
        )

    def test_reject_invalid_bbox_order(self):
        with self.assertRaises(ValueError):
            convert_detections(
                [{"class_id": 0, "confidence": 0.9, "bbox": [300, 85, 120, 410]}],
                ["book_monkey"],
            )


if __name__ == "__main__":
    unittest.main()
