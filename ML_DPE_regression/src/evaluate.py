from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ML_DPE_regression.src.config import get_production_model_path
from ML_DPE_regression.src.features import split_features_target


def evaluate(df: pd.DataFrame) -> dict:
    """
    Évalue le modèle de production sur un dataset externe.
    Retourne un dictionnaire contenant les métriques.
    """

    # -----------------------------
    # Charger le modèle de production
    # -----------------------------
    model_path = get_production_model_path()
    bundle = joblib.load(model_path)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    print(f"📦 Modèle chargé : {model_path}")

    # -----------------------------
    # Séparer X et y
    # -----------------------------
    X, y = split_features_target(df)

    # -----------------------------
    # Appliquer le preprocessing
    # -----------------------------
    X_processed = preprocessor.transform(X)

    # -----------------------------
    # Prédictions
    # -----------------------------
    y_pred = model.predict(X_processed)

    # -----------------------------
    # Calcul des métriques
    # -----------------------------
    metrics = {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": mean_squared_error(y, y_pred, squared=False),
    }

    print("\n📊 ÉVALUATION DU MODÈLE")
    print(f"   R²   : {metrics['r2']:.4f}")
    print(f"   MAE  : {metrics['mae']:.4f}")
    print(f"   RMSE : {metrics['rmse']:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    parser.add_argument("--input", type=str, required=True, help="CSV à évaluer")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";", decimal=",", encoding="utf-8")
    evaluate(df)
