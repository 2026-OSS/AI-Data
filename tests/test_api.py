import unittest

try:
    from fastapi.testclient import TestClient

    from server.api import create_app

    HAS_FASTAPI = True
except ModuleNotFoundError:
    HAS_FASTAPI = False


class FakePredictor:
    def predict(self, image_bytes, filename=None, content_type=None):
        return {
            "page": {"label": "page2", "confidence": 0.9},
            "objects": [
                {
                    "label": "book_monkey",
                    "confidence": 0.8,
                    "bbox": [10, 20, 100, 200],
                }
            ],
            "finger": {"x": 50, "y": 70},
        }


@unittest.skipUnless(HAS_FASTAPI, "fastapi is not installed")
class PredictApiTest(unittest.TestCase):
    def test_health_check(self):
        client = TestClient(create_app(FakePredictor()))

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_returns_backend_response(self):
        client = TestClient(create_app(FakePredictor()))

        response = client.post(
            "/predict",
            files={"frame": ("frame.jpg", b"image-bytes", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "page": {"label": "page2", "confidence": 0.9},
                "objects": [
                    {
                        "label": "book_monkey",
                        "confidence": 0.8,
                        "bbox": [10.0, 20.0, 100.0, 200.0],
                    }
                ],
                "finger": {"x": 50.0, "y": 70.0},
            },
        )

    def test_predict_returns_503_when_predictor_is_not_configured(self):
        client = TestClient(create_app())

        response = client.post(
            "/predict",
            files={"frame": ("frame.jpg", b"image-bytes", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 503)

    def test_predict_rejects_empty_file(self):
        client = TestClient(create_app(FakePredictor()))

        response = client.post(
            "/predict",
            files={"frame": ("frame.jpg", b"", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
