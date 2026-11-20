"""Local inference API; model path is configured by the operator, never clients."""

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .inference import load_model, predict_records


class FeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict[str, Annotated[FiniteFloat, Field(strict=True)]]] = Field(min_length=1, max_length=1000)


def create_app(model_path=None):
    app = FastAPI(title="Cycling Safety ML", version="0.1.0",
                  description="Research demo from archived IMU-window features. Not a safety alert service.")
    path = Path(model_path or os.environ.get("CYCLING_MODEL_PATH", "artifacts/model.json"))
    # Load once at startup/factory time. Missing or malformed models fail closed.
    model = load_model(path)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_type": model["model_type"], "features": len(model["features"])}

    @app.get("/schema")
    def schema():
        return {"feature_names": model["features"], "labels": model["labels"],
                "input": "72 per-window features from 128 samples at 50 Hz", "probabilities_calibrated": False}

    @app.post("/predict")
    def predict(request: FeatureRequest):
        try:
            result = predict_records(model, request.records)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"predictions": result, "notice": "Experimental predictions; not verified events or calibrated risks"}

    return app
