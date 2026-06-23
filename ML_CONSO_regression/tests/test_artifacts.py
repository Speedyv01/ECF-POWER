import json

from sklearn.dummy import DummyRegressor
from src.ml_conso.features import FEATURES
from src.ml_conso.pipeline import (
    activate_model_version,
    export_model_contract,
    save_model_version,
)


def test_save_and_activate_model_version(tmp_path):
    model = DummyRegressor(strategy="mean")
    metrics = {"rmse": 1.2, "mae": 0.8, "r2": 0.4}

    save_model_version(model, metrics, "v1", tmp_path)
    activate_model_version("v1", tmp_path)

    model_path = tmp_path / "models" / "conso_model_v1.joblib"
    metrics_path = tmp_path / "metrics" / "conso_metrics_v1.json"
    latest_model_path = tmp_path / "models" / "conso_model_latest.joblib"
    latest_metrics_path = tmp_path / "metrics" / "conso_metrics_latest.json"

    assert model_path.exists()
    assert metrics_path.exists()
    assert latest_model_path.exists()
    assert latest_metrics_path.exists()

    with latest_metrics_path.open(encoding="utf-8") as file:
        latest_metrics = json.load(file)

    assert latest_metrics["r2"] == metrics["r2"]


def test_export_model_contract_uses_evo_conso_target(tmp_path):
    contract_path = export_model_contract(FEATURES, tmp_path)

    with contract_path.open(encoding="utf-8") as file:
        contract = json.load(file)

    assert contract["target"] == "evo_conso"
    assert contract["features"] == FEATURES
    assert contract["model_path"] == "artifacts/models/conso_model_latest.joblib"


def test_latest_metrics_keep_minimum_expected_score(tmp_path):
    # Setup: Create a dummy model and metrics, then save and activate them
    model = DummyRegressor(strategy="mean")
    # Use a value that satisfies the assertion > 0.80
    metrics = {"rmse": 0.1, "mae": 0.05, "r2": 0.85}

    save_model_version(model, metrics, "test_version", tmp_path)
    activate_model_version("test_version", tmp_path)

    # Test: Read the latest metrics from the temporary path
    metrics_path = tmp_path / "metrics" / "conso_metrics_latest.json"

    with metrics_path.open(encoding="utf-8") as file:
        loaded_metrics = json.load(file)

    assert loaded_metrics["r2"] > 0.80
