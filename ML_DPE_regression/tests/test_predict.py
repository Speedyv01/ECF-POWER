import joblib
from sklearn.dummy import DummyRegressor

import ML_DPE_regression.src.predict as predict_module
from ML_DPE_regression.src.features import split_features_target
from ML_DPE_regression.src.preprocessing import ProductionPreprocessor


def test_predict_loads_modele_pkl_bundle(sample_df, workspace_tmp_path, monkeypatch):
    X, y = split_features_target(sample_df)
    preprocessor = ProductionPreprocessor()
    X_processed = preprocessor.fit_transform(X)

    model = DummyRegressor(strategy="mean")
    model.fit(X_processed, y)

    bundle_path = workspace_tmp_path / "modele.pkl"
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "metadata": {"best_model": "dummy"},
        },
        bundle_path,
    )
    monkeypatch.setattr(
        predict_module, "get_production_model_path", lambda: bundle_path
    )

    y_pred, metadata = predict_module.predict(sample_df)

    assert len(y_pred) == len(sample_df)
    assert metadata["best_model"] == "dummy"
