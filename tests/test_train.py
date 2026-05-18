import json


import src.train as train_module


def test_train_creates_production_bundle(sample_df, workspace_tmp_path, monkeypatch):
    # Override the output directories to use the test workspace
    monkeypatch.setattr("src.train.METRICS_DIR", workspace_tmp_path)
    monkeypatch.setattr(
        "src.train.get_production_model_path",
        lambda: workspace_tmp_path / "modele.pkl",
    )
    monkeypatch.setattr(
        "src.train.get_latest_model_info_path",
        lambda: workspace_tmp_path / "model.latest.json",
    )

    model, preprocessor, metadata = train_module.train(sample_df)

    assert model is not None
    assert preprocessor is not None
    assert metadata["best_model"] == "random_forest"
    assert (workspace_tmp_path / "modele.pkl").exists()
    assert (workspace_tmp_path / "model.latest.json").exists()

    latest = json.loads(
        (workspace_tmp_path / "model.latest.json").read_text(encoding="utf-8")
    )
    assert latest["model_name"] == "random_forest"
