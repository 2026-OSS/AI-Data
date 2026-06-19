import unittest

from fastapi.testclient import TestClient

from server.api import create_app


class BackendContractPredictor:
    def predict(self, image_bytes, filename=None, content_type=None):
        return {
            "page": {"label": "page2", "confidence": 0.91},
            "objects": [
                {
                    "label": "book_monkey",
                    "confidence": 0.84,
                    "bbox": [120, 85, 300, 410],
                },
                {
                    "label": "text",
                    "confidence": 0.77,
                    "bbox": [20, 40, 360, 120],
                },
            ],
            "finger": {"x": 210, "y": 180},
        }


class BackendIntegrationTest(unittest.TestCase):
    def test_predict_accepts_backend_frame_field(self):
        client = TestClient(create_app(BackendContractPredictor()))

        response = client.post(
            "/predict",
            files={"frame": ("frame.jpg", b"image-bytes", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)

    def test_predict_response_matches_backend_contract(self):
        client = TestClient(create_app(BackendContractPredictor()))

        response = client.post(
            "/predict",
            files={"frame": ("frame.jpg", b"image-bytes", "image/jpeg")},
        )
        body = response.json()

        self.assertEqual(set(body.keys()), {"page", "objects", "finger"})
        self.assertEqual(set(body["page"].keys()), {"label", "confidence"})
        self.assertEqual(set(body["objects"][0].keys()), {"label", "confidence", "bbox"})
        self.assertEqual(set(body["finger"].keys()), {"x", "y"})
        self.assertIsInstance(body["page"]["label"], str)
        self.assertIsInstance(body["page"]["confidence"], float)
        self.assertIsInstance(body["objects"][0]["label"], str)
        self.assertEqual(len(body["objects"][0]["bbox"]), 4)
        self.assertIsInstance(body["finger"]["x"], float)
        self.assertIsInstance(body["finger"]["y"], float)

    def test_predict_rejects_wrong_file_field(self):
        client = TestClient(create_app(BackendContractPredictor()))

        response = client.post(
            "/predict",
            files={"image": ("frame.jpg", b"image-bytes", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
