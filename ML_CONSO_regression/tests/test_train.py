from sklearn.ensemble import RandomForestRegressor

from src.ml_conso.train import build_model


def test_build_model_returns_configured_random_forest():
    model = build_model()

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 200
    assert model.max_depth == 20
    assert model.min_samples_split == 5
    assert model.random_state == 42
