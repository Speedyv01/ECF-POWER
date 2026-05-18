import numpy as np

from src.ml_conso.features import FEATURES, split_data
from src.ml_conso.preprocessing import build_preprocessor


def test_build_preprocessor_transforms_current_features(complete_selected_df):
    sample = complete_selected_df.head(100)
    X_train, _, y_train, _ = split_data(sample)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_train, y_train)

    assert transformed.shape[0] == len(X_train)
    assert transformed.shape[1] == len(FEATURES)


def test_build_preprocessor_imputes_missing_values(complete_selected_df):
    sample = complete_selected_df.head(100).copy()
    sample.loc[sample.index[0], "TEMP_MAX_SA"] = None
    X_train, _, y_train, _ = split_data(sample)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(X_train, y_train)

    assert not np.isnan(transformed).any()
