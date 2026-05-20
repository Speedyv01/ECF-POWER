from __future__ import annotations

import joblib
import pandas as pd

from ML_DPE_regression.src.config import get_production_model_path

# ============================================================
# 1. Chargement du bundle
# ============================================================


def load_bundle():
    """
    Charge le bundle complet (modèle + préprocesseur + metadata)
    depuis artifacts/models/modele.pkl.
    """
    model_path = get_production_model_path()
    bundle = joblib.load(model_path)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    metadata = bundle.get("metadata", {})

    return model, preprocessor, metadata


# ============================================================
# 2. Prédiction
# ============================================================


def predict(df: pd.DataFrame):
    """
    Applique le préprocesseur + modèle pour produire une prédiction.
    df doit contenir les colonnes brutes (non prétraitées).
    """

    # Charger modèle + préprocesseur
    model, preprocessor, metadata = load_bundle()

    # Préprocessing
    X_processed = preprocessor.transform(df)

    # Prédiction
    y_pred = model.predict(X_processed)

    return y_pred, metadata


# ============================================================
# 3. Exemple d'utilisation
# ============================================================

if __name__ == "__main__":
    # Exemple : prédire à partir d'un fichier CSV
    df = pd.read_csv("input.csv")
    y_pred, metadata = predict(df)

    print("🔮 Prédictions :", y_pred)
    print("ℹ️  Modèle utilisé :", metadata.get("best_model"))
