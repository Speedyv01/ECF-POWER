import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from api.backend.call_weather_api import get_forecast, merge_weather_sun

# --- 1. Gestion des Chemins et Imports ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
ml_conso_root = project_root / "ML_CONSO_regression"
ml_dpe_root = project_root / "ML_DPE_regression"
ml_conso_src_path = ml_conso_root / "src"
ml_dpe_src_path = ml_dpe_root / "src"

# `src` is a top-level package in both ML projects. We only add the relevant
# project root to sys.path when loading each model to avoid package conflicts.


""" Bloc utilisé lors du test, à supprimer si pas de PP séparé du modèle
try:
    from preprocessing_utils import preprocess_ecs_column

    print("✅ Module de preprocessing chargé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
"""

app = FastAPI()

# --- 2. Fonctions Utilitaires ---


def _load_model_with_root(model_path: Path, root_path: Path):
    original_sys_path = sys.path.copy()
    # Remove any previously loaded `src` package from other ML projects, because
    # both DPE and CONSO use a top-level package named `src`.
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            del sys.modules[module_name]

    try:
        root_str = str(root_path)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        return joblib.load(model_path)
    finally:
        sys.path[:] = original_sys_path


def get_latest_model(cat: str):
    """Charge le fichier modèle s'il existe selon la catégorie."""
    if cat == "conso":
        model_name = "conso_model_latest.joblib"
        latest_model_path = ml_conso_root / "artifacts" / "models" / model_name
        loader_root = ml_conso_root
    elif cat == "DPE":
        model_name = "DPE_model_latest.pkl"
        latest_model_path = ml_dpe_root / "artifacts" / "models" / model_name
        loader_root = ml_dpe_root
    else:
        raise ValueError(f"Catégorie inconnue : {cat}")

    if not latest_model_path.exists():
        raise FileNotFoundError(
            f"❌ Aucun modèle trouvé pour {cat} à : {latest_model_path}"
        )

    loaded = _load_model_with_root(latest_model_path, loader_root)

    if cat == "DPE" and isinstance(loaded, dict):
        return loaded
    return loaded


def predict(cat: str, data: dict):
    """Effectue une prédiction et renvoie un dictionnaire."""
    model_obj = get_latest_model(cat)
    df = pd.DataFrame([data])
    
    if cat == "DPE" and isinstance(model_obj, dict):
        # DPE: extract model and preprocessor
        preprocessor = model_obj.get("preprocessor")
        model = model_obj.get("model")
        # Preprocess the data
        df_processed = preprocessor.transform(df)
        prediction = model.predict(df_processed)[0]
    else:
        # CONSO: direct prediction (pipeline handles preprocessing)
        prediction = model_obj.predict(df)[0]
    
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
        # Si le user n'a pas fait d'estimation, on met à zéro
        # pour éviter les erreurs de calcul
        conso_jour = 1
    else:
        conso_jour = data["conso_jour"]

    # 1. Récupération prévis météo + durée ensoleillement
    forecast = get_forecast(data["zipcode"], data["api_key"])
    city_name = forecast["city"]["name"]
    meteo = merge_weather_sun(forecast)

    # 2. Prédictions itératives
    # On Transforme le dataframe en dictionnaire pour le parcourir
    meteo_dict_list = []
    for _, row in meteo.iterrows():
        ligne = row.to_dict()
        # Prediction désactivé pour les tests
        res_predict = predict("conso", ligne)
        #res_predict = {"prediction": 1}  # Ligne à supprimer en prod
        ligne["Conso_Estimee"] = res_predict["prediction"] * conso_jour

        meteo_dict_list.append(ligne)

    # 3. Préparation de la réponse
    df_result = pd.DataFrame(meteo_dict_list)
    df_result = df_result.rename(
        columns={
            "date": "Date",
            "sunrise": "Lever 🌅",
            "sunset": "Coucher 🌇",
            "MOYENNE_TEMP_HORAIRES_SA_PONDEREE": "T° moyenne",
            "MOYENNE_HUMIDITES_RELATIVES_HORAIRES": "Humidité moyenne",
            "TEMP_MIN_SOUS_ABRI": "T° min",
            "TEMP_MAX_SA": "T° max",
            "Conso_Estimee": "Conso",
        }
    )
    # On formate les colonnes numériques à 2 décimales pour l'affichage
    columns_to_round = ["T° moyenne", "Humidité moyenne", "T° min", "T° max", "Conso"]
    for col in columns_to_round:
        df_result[col] = df_result[col].apply(
            lambda x: round(x, 2) if pd.notnull(x) else None
        )
    # On a toujours la colonne durée dans notre df mais on ne l'affichera pas

    return {"city_name": city_name, "predictions": df_result.to_dict(orient="records")}
