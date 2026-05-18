import pytest

from src.features import (
    TARGET_COLUMN,
    TOP_FEATURES,
    select_features,
    select_target,
    validate_training_columns,
)


def test_feature_configuration_uses_expected_target(sample_df):
    assert TARGET_COLUMN == "conso_5_usages_ep"
    assert len(TOP_FEATURES) == 6
    validate_training_columns(sample_df)


def test_select_features_and_target(sample_df):
    X = select_features(sample_df)
    y = select_target(sample_df)

    assert list(X.columns) == TOP_FEATURES
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y)


def test_validate_training_columns_reports_missing_column(sample_df):
    df = sample_df.drop(columns=[TOP_FEATURES[0]])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_training_columns(df)
