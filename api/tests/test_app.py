import sys
from pathlib import Path

current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.backend import app  # noqa: E402


def test_index_returns_message():
    client = TestClient(app.app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "API POWER",
        "docs": "/docs",
        "santé": "/health",
        "résultats entraînement": "/ml",
    }


def test_health_returns_ok():
    client = TestClient(app.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_conso_uses_forecast_and_returns_predictions(monkeypatch):
    client = TestClient(app.app)

    def fake_get_forecast(zipcode, api_key):
        return {"city": {"name": "Paris"}}

    def fake_merge_weather_sun(forecast):
        return pd.DataFrame(
            [
                {
                    "date": "2026-05-19",
                    "sunrise": "06:00",
                    "sunset": "21:00",
                    "MOYENNE_TEMP_HORAIRES_SA_PONDEREE": 20.12,
                    "MOYENNE_HUMIDITES_RELATIVES_HORAIRES": 55.5,
                    "TEMP_MIN_SOUS_ABRI": 15.0,
                    "TEMP_MAX_SA": 25.0,
                    "DUREE_ENSOLEILLEMENT": 8.5,
                }
            ]
        )

    def fake_predict(cat, data):
        assert cat == "conso"
        assert "MOYENNE_TEMP_HORAIRES_SA_PONDEREE" in data
        return {"prediction": 1.0}

    monkeypatch.setattr(app, "get_forecast", fake_get_forecast)
    monkeypatch.setattr(app, "merge_weather_sun", fake_merge_weather_sun)
    monkeypatch.setattr(app, "predict", fake_predict)

    response = client.post(
        "/predict_conso",
        json={"zipcode": "75014", "api_key": "fake_key", "conso_jour": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["city_name"] == "Paris"
    assert isinstance(body["predictions"], list)
    assert body["predictions"][0]["Conso"] == 0.0


def test_ml_returns_model_results(monkeypatch):
    client = TestClient(app.app)

    monkeypatch.setattr(app, "get_model_results", lambda cat: {"metric": 1.0})

    response = client.get("/ml")

    assert response.status_code == 200
    assert response.json() == {
        "Résultats de l'entraînement du modèle conso": {"metric": 1.0},
        "Résultats de l'entraînement du modèle DPE": {"metric": 1.0},
    }
