from __future__ import annotations

import inspect
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from server.backend_response import build_backend_response
from server.predictor import PredictionUnavailable, Predictor, UnconfiguredPredictor


def create_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="AI-Data Predict API")
    app.state.predictor = predictor or UnconfiguredPredictor()

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(frame: UploadFile = File(...)) -> dict[str, Any]:
        image_bytes = await frame.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="frame file is empty",
            )

        try:
            raw_response = app.state.predictor.predict(
                image_bytes=image_bytes,
                filename=frame.filename,
                content_type=frame.content_type,
            )
            if inspect.isawaitable(raw_response):
                raw_response = await raw_response
            return build_backend_response(
                page=raw_response["page"],
                objects=raw_response.get("objects", []),
                finger=raw_response.get("finger"),
            )
        except PredictionUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    return app


app = create_app()
