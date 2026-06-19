from __future__ import annotations

from server.api import create_app
from server.model_predictor import create_default_predictor
from server.predictor import PredictionUnavailable, UnconfiguredPredictor


try:
    predictor = create_default_predictor()
except PredictionUnavailable:
    predictor = UnconfiguredPredictor()


app = create_app(predictor)
