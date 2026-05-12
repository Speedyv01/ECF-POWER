import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from api.backend.call_weather_api import get_forecast, merge_weather_sun

# --- 1. Gestion des Chemins et Imports ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
ml_src_path = project_root / "ml" / "src"

sys.path.append(str(ml_src_path))

try:
    from preprocessing_utils import preprocess_ecs_column

    print("✅ Module de preprocessing chargé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")

app = FastAPI()

# --- 2. Fonctions Utilitaires ---


def get_latest_model(cat: str):
    """Charge le fichier (cat)_model_latest.pkl s'il existe."""
    model_name = f"{cat}_model_latest.pkl"
    latest_model_path = project_root / "ml" / "models" / model_name

    # Correction de l'erreur : On vérifie si le CHEMIN existe avant de charger
    if not latest_model_path.exists():
        raise FileNotFoundError(
            f"❌ Aucun modèle trouvé pour {cat} à : {latest_model_path}"
        )

    return joblib.load(latest_model_path)


def predict(cat: str, data: dict):
    """Effectue une prédiction et renvoie un dictionnaire."""
    model = get_latest_model(cat)
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}


# --- 3. Points d'Entrée (Endpoints) ---


@app.get("/")
def index():
    return {"message": "API POWER", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_DPE")
def predict_dpe(data: dict):
    return predict("DPE", data)


@app.post("/predict_conso")
def predict_conso(data: dict):
    if not data["conso_jour"]:
        conso_jour = 0
    else:
        conso_jour = data["conso_jour"]

    # 1. Récupération météo
    forecast = get_forecast(data["zipcode"], data["api_key"])
    city_name = forecast["city"]["name"]
    meteo = merge_weather_sun(forecast)

    # 2. Prédictions itératives
    meteo_dict_list = []
    for _, row in meteo.iterrows():
        ligne = row.to_dict()
        # res_predict = predict("Conso", ligne)
        res_predict = {"prediction": 1}
        ligne["Conso_Estimee"] = res_predict["prediction"] * conso_jour

        meteo_dict_list.append(ligne)

    # 3. Préparation de la réponse
    df_result = pd.DataFrame(meteo_dict_list)
    df_result = df_result.rename(
        columns={
            "date": "Date",
            "sunrise": "Lever 🌅",
            "sunset": "Coucher 🌇",
            "MOYENNE_TEMP_HORAIRES_SA": "T° moyenne",
            "MOYENNE_HUMIDITES_RELATIVES_HORAIRES": "Humidité moyenne",
            "TEMP_MIN_SOUS_ABRI": "T° min",
            "TEMP_MAX_SA": "T° max",
            "Conso_Estimee": "Conso"
        }
    )

    return {"city_name": city_name, "predictions": df_result.to_dict(orient="records")}
