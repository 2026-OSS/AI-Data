from __future__ import annotations

from typing import Any, Protocol


class PredictionUnavailable(RuntimeError):
    pass


class Predictor(Protocol):
    def predict(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        ...


class UnconfiguredPredictor:
    def predict(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        raise PredictionUnavailable("prediction pipeline is not configured")
