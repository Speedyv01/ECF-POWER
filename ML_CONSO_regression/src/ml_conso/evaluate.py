import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    return metrics


def compute_metrics(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


def compute_feature_importance(model, X, y, feature_names):
    if not hasattr(model, "feature_importances_"):
        model.fit(X, y)

    return dict(zip(feature_names, model.feature_importances_))
