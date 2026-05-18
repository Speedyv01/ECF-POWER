from src.ml_conso.features import FEATURES, split_data
from src.ml_conso.pipeline import build_pipeline, run_pipeline


def test_build_pipeline_has_preprocessing_then_model():
    pipeline = build_pipeline()

    assert list(pipeline.named_steps) == ["preprocessing", "model"]


def test_pipeline_can_fit_and_predict_on_small_sample(complete_selected_df):
    sample = complete_selected_df.head(300)
    X_train, X_test, y_train, _ = split_data(sample)
    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)


def test_run_pipeline_returns_expected_contract(monkeypatch, raw_df):
    required_columns = ["Date", "Conso_MWH"] + FEATURES
    sample = raw_df.dropna(subset=required_columns).head(300)
    monkeypatch.setattr("src.ml_conso.pipeline.load_data", lambda: sample.copy())

    results = run_pipeline()

    assert set(results) == {"best_model", "evaluation", "feature_names"}
    assert results["feature_names"] == [
        "DUREE_ENSOLEILLEMENT",
        "MOYENNE_TEMP_HORAIRES_SA_PONDEREE",
        "TEMP_MAX_SA",
        "MOYENNE_HUMIDITES_RELATIVES_HORAIRES",
        "TEMP_MIN_SOUS_ABRI",
    ]
    assert set(results["evaluation"]) == {"rmse", "mae", "r2"}
