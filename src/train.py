from __future__ import annotations

import json
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config import (
    get_production_model_path,
    get_latest_model_info_path,
    METRICS_DIR,
)
from src.preprocessing import preprocess_pipeline
from src.features import split_features_target

# ============================================================
# 1. Extraction des features importantes
# ============================================================


def extract_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    return {name: None for name in feature_names}


# ============================================================
# 2. Entraînement RandomForest
# ============================================================


def train(df):

    # -----------------------------
    # Sélection des features
    # -----------------------------
    X, y = split_features_target(df)

    print("\n=== Features utilisées ===")
    print(X.columns.tolist())
    print(f"Nombre de features : {len(X.columns)}")

    # 🔥 Correction : imputation du y
    y = y.fillna(y.median())

    # -----------------------------
    # Préprocessing
    # -----------------------------
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(X, y)

    # -----------------------------
    # Modèle RandomForest
    # -----------------------------
    print("\n🌲 Entraînement du RandomForest...")

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

    print(f"   → R²   = {metrics['r2']:.4f}")
    print(f"     MAE  = {metrics['mae']:.4f}")
    print(f"     RMSE = {metrics['rmse']:.4f}")

    # -----------------------------
    # Feature importances
    # -----------------------------
    feature_importances = extract_feature_importances(model, preprocessor.feature_names)

    feature_importances_path = METRICS_DIR / "feature_importances.json"
    with open(feature_importances_path, "w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=4, ensure_ascii=False)

    print(f"📊 Feature importances sauvegardées dans : {feature_importances_path}")

    # -----------------------------
    # Sauvegarde du bundle complet
    # -----------------------------
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": {
            "best_model": "random_forest",
            "best_metrics": metrics,
            "all_metrics": {"random_forest": metrics},
            "feature_importances_path": str(feature_importances_path),
        },
    }

    model_path = get_production_model_path()
    joblib.dump(bundle, model_path)

    print(f"💾 Modèle sauvegardé dans : {model_path}")

    # -----------------------------
    # Sauvegarde des infos du modèle
    # -----------------------------
    latest_info_path = get_latest_model_info_path()
    with open(latest_info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "random_forest",
                "model_path": str(model_path),
                "best_metrics": metrics,
                "all_metrics": {"random_forest": metrics},
                "feature_importances": feature_importances,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"📄 Infos sauvegardées dans : {latest_info_path}")

    return (
        model,
        preprocessor,
        {
            "best_model": "random_forest",
            "best_metrics": metrics,
            "all_metrics": {"random_forest": metrics},
        },
    )
