import json
from pathlib import Path

from fastapi.testclient import TestClient

from cycling_safety.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_actual_deployment_model_with_shipped_request():
    request = json.loads((ROOT / "examples/example_request.json").read_text())
    with TestClient(create_app(ROOT / "artifacts/model.json")) as client:
        response = client.post("/predict", json=request)
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 4
        assert all(len(item["probabilities"]) == 4 for item in predictions)


def test_saved_results_and_nested_boundaries_are_auditable():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_results", ROOT / "scripts/verify_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify(ROOT)["status"] == "passed"
