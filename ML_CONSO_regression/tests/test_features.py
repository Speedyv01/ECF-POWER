from src.ml_conso.features import FEATURES, TARGET, split_data


def test_select_features_keeps_current_model_columns(selected_df):
    expected_columns = FEATURES + [TARGET]

    assert selected_df.columns.tolist() == expected_columns


def test_evo_conso_is_current_target(selected_df):
    assert TARGET == "evo_conso"
    assert TARGET in selected_df.columns
    assert selected_df[TARGET].notna().all()


def test_split_data_uses_evo_conso_as_y(selected_df):
    X_train, X_test, y_train, y_test = split_data(selected_df)

    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns
    assert y_train.name == TARGET
    assert y_test.name == TARGET
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_selected_features_are_numeric_inputs(selected_df):
    assert selected_df[FEATURES].select_dtypes(include=["number"]).shape[1] == len(
        FEATURES
    )
