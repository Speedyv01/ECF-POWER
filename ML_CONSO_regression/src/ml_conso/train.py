import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.ml_conso.evaluate import evaluate_model


def build_model():

    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=1
    )

    return model


def log_experiment(metrics: dict):

    mlflow.log_params(
        {
            "model": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
        }
    )

    mlflow.log_metrics(metrics)


def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_base_models(X_train, y_train, X_test, y_test):
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=1
        ),
        "LinearRegression": LinearRegression(),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {
            "model": model,
            "rmse_test": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }

    return results


def select_best_model(results):
    best_name = max(results, key=lambda name: results[name]["r2"])
    return best_name, results[best_name]["model"]
