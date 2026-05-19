from sklearn.dummy import DummyRegressor
from src.ml_conso.evaluate import evaluate_model
from src.ml_conso.features import split_data


def test_evaluate_model_returns_expected_metrics(selected_df):
    X_train, X_test, y_train, y_test = split_data(selected_df)
    model = DummyRegressor(strategy="mean")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())
