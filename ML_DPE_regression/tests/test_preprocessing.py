import numpy as np

from ML_DPE_regression.src.features import TOP_FEATURES, split_features_target
from ML_DPE_regression.src.preprocessing import (
    ProductionPreprocessor,
    preprocess_pipeline,
)


def test_production_preprocessor_returns_numeric_dataframe(sample_df):
    X, _ = split_features_target(sample_df)
    preprocessor = ProductionPreprocessor()

    transformed = preprocessor.fit_transform(X)

    assert list(transformed.columns) == TOP_FEATURES
    assert transformed.shape == X.shape
    assert np.isfinite(transformed.to_numpy()).all()


def test_preprocess_pipeline_splits_and_transforms(sample_df):
    X, y = split_features_target(sample_df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        X,
        y,
        remove_outliers=False,
    )

    assert X_train.shape[1] == len(TOP_FEATURES)
    assert X_test.shape[1] == len(TOP_FEATURES)
    assert len(y_train) + len(y_test) == len(sample_df)
    assert isinstance(preprocessor, ProductionPreprocessor)
