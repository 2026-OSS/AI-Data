import unittest

from server.backend_response import (
    build_backend_response,
    normalize_finger,
    normalize_object,
    normalize_page,
)


class BackendResponseTest(unittest.TestCase):
    def test_builds_backend_response(self):
        response = build_backend_response(
            page={"label": "page2", "confidence": 0.93},
            objects=[
                {
                    "label": "book_monkey",
                    "confidence": 0.86,
                    "bbox": [10, 20, 110, 220],
                }
            ],
            finger={"x": 50, "y": 70},
        )

        self.assertEqual(
            response,
            {
                "page": {"label": "page2", "confidence": 0.93},
                "objects": [
                    {
                        "label": "book_monkey",
                        "confidence": 0.86,
                        "bbox": [10.0, 20.0, 110.0, 220.0],
                    }
                ],
                "finger": {"x": 50.0, "y": 70.0},
            },
        )

    def test_allows_no_finger(self):
        response = build_backend_response(
            page={"label": "page1", "confidence": 1},
            objects=[],
            finger=None,
        )

        self.assertIsNone(response["finger"])

    def test_accepts_compatible_aliases(self):
        page = normalize_page({"page": "page3", "confidence": 0.9})
        detected = normalize_object(
            {"class_name": "text", "confidence": 0.8, "bbox": [1, 2, 3, 4]}
        )

        self.assertEqual(page["label"], "page3")
        self.assertEqual(detected["label"], "text")

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            normalize_page({"label": "page1", "confidence": 1.2})

    def test_rejects_invalid_bbox(self):
        with self.assertRaises(ValueError):
            normalize_object({"label": "book_monkey", "confidence": 0.9, "bbox": [1]})

    def test_rejects_invalid_finger(self):
        with self.assertRaises(ValueError):
            normalize_finger({"x": 10})


if __name__ == "__main__":
    unittest.main()
