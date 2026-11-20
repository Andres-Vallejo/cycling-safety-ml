import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cycling_safety.api import create_app
from cycling_safety.contract import FEATURE_NAMES, features_from_window
from cycling_safety.data import write_json
from cycling_safety.evaluation import fit_model
from cycling_safety.inference import export_mlp, load_model, predict_proba, predict_records


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 72))
    y = np.tile(np.arange(4), 25)
    model, _ = fit_model("mlp", {"hidden_layer_sizes": [8], "alpha": 0.01}, 7, 10, x, y)
    return x, model, export_mlp(model, "fixture", len(x), 5)


def test_json_model_roundtrip_matches_sklearn(trained, tmp_path):
    x, model, portable = trained
    write_json(tmp_path / "model.json", portable)
    loaded = load_model(tmp_path / "model.json")
    np.testing.assert_allclose(predict_proba(loaded, x), model.predict_proba(x), rtol=1e-10, atol=1e-11)


def test_feature_order_is_named_and_metadata_is_rejected(trained):
    x, _, portable = trained
    record = dict(zip(reversed(FEATURE_NAMES), reversed(x[0].tolist())))
    actual = predict_records(portable, [record])[0]["probabilities"]
    np.testing.assert_allclose(list(actual.values()), predict_proba(portable, x[:1])[0])
    record["label"] = 1
    with pytest.raises(ValueError, match="exactly"):
        predict_records(portable, [record])


def test_nonfinite_inputs_and_broken_model_are_rejected(trained):
    x, _, portable = trained
    values = x[:1].copy()
    values[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predict_proba(portable, values)
    broken = {**portable, "scale": [0] * 72}
    with pytest.raises(ValueError, match="scale"):
        predict_proba(broken, x[:1])


def test_api_predict_and_validation(trained, tmp_path):
    x, _, portable = trained
    write_json(tmp_path / "model.json", portable)
    with TestClient(create_app(tmp_path / "model.json")) as client:
        assert client.get("/health").json()["features"] == 72
        assert client.get("/schema").json()["probabilities_calibrated"] is False
        record = dict(zip(FEATURE_NAMES, x[0].tolist()))
        response = client.post("/predict", json={"records": [record]})
        assert response.status_code == 200
        probabilities = response.json()["predictions"][0]["probabilities"]
        assert sum(probabilities.values()) == pytest.approx(1)
        assert client.post("/predict", json={"records": []}).status_code == 422
        assert client.post("/predict", json={"records": [{"ax_mean": 0}]}).status_code == 422
        assert client.post("/predict", json={"records": [record], "model_path": "other"}).status_code == 422
        assert client.post("/predict", json={"records": [{**record, "ax_mean": True}]}).status_code == 422
        assert client.post("/predict", json={"records": [{**record, "ax_mean": "oops"}]}).status_code == 422


def test_cli_prediction(trained, tmp_path):
    x, _, portable = trained
    write_json(tmp_path / "model.json", portable)
    pd.DataFrame(x[:2], columns=FEATURE_NAMES).to_csv(tmp_path / "input.csv", index=False)
    result = subprocess.run([sys.executable, "-m", "cycling_safety.cli", "predict", "--model",
        str(tmp_path / "model.json"), "--csv", str(tmp_path / "input.csv"), "--output", str(tmp_path / "result.json")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads((tmp_path / "result.json").read_text())["predictions"]) == 2


def test_extractor_matches_archived_upstream_reference():
    fixture = json.loads((Path(__file__).parent / "fixtures/feature_reference.json").read_text())
    actual = features_from_window(fixture["samples"])
    np.testing.assert_allclose(list(actual.values()), fixture["expected"], rtol=2e-6, atol=1e-7)


def test_constant_and_malformed_windows():
    values = features_from_window(np.zeros((128, 6)))
    assert len(values) == 72 and set(values.values()) == {0.0}
    with pytest.raises(ValueError, match="128 x 6"):
        features_from_window(np.zeros((120, 6)))
