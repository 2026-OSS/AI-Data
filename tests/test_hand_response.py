import unittest
from types import SimpleNamespace

from server.hand_response import (
    convert_normalized_point,
    extract_finger_from_results,
    extract_index_finger_tip,
)


class HandResponseTest(unittest.TestCase):
    def test_converts_normalized_point_to_pixel_point(self):
        point = convert_normalized_point(0.5, 0.25, 640, 480)

        self.assertEqual(point, {"x": 320.0, "y": 120.0})

    def test_extracts_index_finger_tip_from_hand_landmarks(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
        landmarks[8] = SimpleNamespace(x=0.25, y=0.75)
        hand_landmarks = SimpleNamespace(landmark=landmarks)

        finger = extract_index_finger_tip(hand_landmarks, 800, 600)

        self.assertEqual(finger, {"x": 200.0, "y": 450.0})

    def test_extracts_first_hand_from_mediapipe_results(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
        landmarks[8] = SimpleNamespace(x=0.1, y=0.2)
        results = SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)]
        )

        finger = extract_finger_from_results(results, 1000, 500)

        self.assertEqual(finger, {"x": 100.0, "y": 100.0})

    def test_extracts_first_hand_from_mediapipe_tasks_results(self):
        landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
        landmarks[8] = SimpleNamespace(x=0.3, y=0.4)
        results = SimpleNamespace(hand_landmarks=[landmarks])

        finger = extract_finger_from_results(results, 1000, 500)

        self.assertEqual(finger, {"x": 300.0, "y": 200.0})

    def test_returns_none_when_hand_is_not_detected(self):
        results = SimpleNamespace(multi_hand_landmarks=[])

        finger = extract_finger_from_results(results, 640, 480)

        self.assertIsNone(finger)

    def test_rejects_invalid_image_size(self):
        with self.assertRaises(ValueError):
            convert_normalized_point(0.5, 0.5, 0, 480)

    def test_rejects_missing_index_finger_tip(self):
        hand_landmarks = SimpleNamespace(landmark=[])

        with self.assertRaises(ValueError):
            extract_index_finger_tip(hand_landmarks, 640, 480)


if __name__ == "__main__":
    unittest.main()
