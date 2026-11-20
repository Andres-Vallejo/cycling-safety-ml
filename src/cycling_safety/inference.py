"""Portable MLP inference from numeric JSON, without loading pickle artifacts."""

import json
from pathlib import Path

import numpy as np

from .contract import FEATURE_NAMES, LABEL_NAMES, validate_matrix


def export_mlp(pipeline, data_sha256, windows, recordings):
    scale, model = pipeline.named_steps["scale"], pipeline.named_steps["model"]
    if model.activation != "relu" or model.out_activation_ != "softmax" or list(model.classes_) != list(range(4)):
        raise ValueError("Only the documented four-class ReLU MLP can be exported")
    return {"schema_version": 1, "model_type": "standard_scaler_relu_mlp_softmax",
            "features": list(FEATURE_NAMES), "labels": list(LABEL_NAMES),
            "mean": scale.mean_.tolist(), "scale": scale.scale_.tolist(),
            "weights": [v.tolist() for v in model.coefs_], "biases": [v.tolist() for v in model.intercepts_],
            "training": {"data_sha256": data_sha256, "windows": windows, "recordings": recordings,
                         "purpose": "All-data refit for demonstration; no independent final-model test score"}}


def load_model(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("schema_version") != 1 or model.get("model_type") != "standard_scaler_relu_mlp_softmax":
        raise ValueError("Unsupported model format")
    if model.get("features") != list(FEATURE_NAMES) or model.get("labels") != list(LABEL_NAMES):
        raise ValueError("Model feature/label contract mismatch")
    # Validate dimensions and numeric values before serving.
    predict_proba(model, np.zeros((1, len(FEATURE_NAMES))))
    return model


def predict_proba(model, values):
    x = validate_matrix(values)
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    if mean.shape != (72,) or scale.shape != (72,) or np.any(scale <= 0):
        raise ValueError("Invalid scaler dimensions or scale")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Invalid scaler values")
    weights, biases = model["weights"], model["biases"]
    if not weights or len(weights) != len(biases):
        raise ValueError("Invalid model layers")
    activation = (x - mean) / scale
    with np.errstate(over="raise", invalid="raise"):
        for layer, (weight, bias) in enumerate(zip(weights, biases)):
            w, b = np.asarray(weight), np.asarray(bias)
            if w.ndim != 2 or w.shape[0] != activation.shape[1] or b.shape != (w.shape[1],):
                raise ValueError("Invalid layer dimensions")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("Invalid layer values")
            activation = activation @ w + b
            if layer < len(weights) - 1:
                activation = np.maximum(activation, 0)
        if activation.shape[1] != 4:
            raise ValueError("Expected four output classes")
        logits = activation - activation.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite predictions")
    return probabilities


def predict_records(model, records):
    if not records or len(records) > 1000:
        raise ValueError("Provide between 1 and 1000 feature records")
    values = []
    for record in records:
        if set(record) != set(FEATURE_NAMES):
            raise ValueError("Each record must contain exactly the 72 documented features")
        row = [record[key] for key in FEATURE_NAMES]
        if any(isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)) for value in row):
            raise ValueError("Feature values must be numbers, not strings or booleans")
        values.append(row)
    probabilities = predict_proba(model, values)
    return [{"label": LABEL_NAMES[int(np.argmax(row))],
             "probabilities": {name: float(p) for name, p in zip(LABEL_NAMES, row)}} for row in probabilities]
