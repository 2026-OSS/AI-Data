import unittest

from server.page_response import (
    DEFAULT_PAGE_CLASSES,
    convert_page_label,
    convert_page_prediction,
)


class ArrayLikeProbabilities:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __bool__(self):
        raise ValueError("truth value is ambiguous")


class PageResponseTest(unittest.TestCase):
    def test_default_page_class_order(self):
        self.assertEqual(DEFAULT_PAGE_CLASSES, ["none", "page1", "page2", "page3"])

    def test_convert_page_prediction_to_backend_page(self):
        page = convert_page_prediction([0.05, 0.1, 0.8, 0.05])

        self.assertEqual(page["label"], "page2")
        self.assertEqual(page["confidence"], 0.8)
        self.assertAlmostEqual(page["margin"], 0.7)
        self.assertEqual(
            page["top_k"],
            [
                {"label": "page2", "confidence": 0.8},
                {"label": "page1", "confidence": 0.1},
                {"label": "none", "confidence": 0.05},
                {"label": "page3", "confidence": 0.05},
            ],
        )

    def test_convert_array_like_page_prediction_to_backend_page(self):
        page = convert_page_prediction(
            ArrayLikeProbabilities([0.05, 0.1, 0.8, 0.05])
        )

        self.assertEqual(page["label"], "page2")
        self.assertEqual(page["confidence"], 0.8)
        self.assertEqual(page["top_k"][0], {"label": "page2", "confidence": 0.8})

    def test_convert_page_label_to_backend_page(self):
        page = convert_page_label("page3", 0.91)

        self.assertEqual(page, {"label": "page3", "confidence": 0.91})

    def test_reject_unknown_page_label(self):
        with self.assertRaises(ValueError):
            convert_page_label("page4", 0.9)

    def test_reject_mismatched_probability_count(self):
        with self.assertRaises(ValueError):
            convert_page_prediction([0.1, 0.9], ["none", "page1", "page2"])


if __name__ == "__main__":
    unittest.main()
