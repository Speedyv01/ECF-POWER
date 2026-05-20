from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREPROCESSORS_DIR,
)

# -----------------------------
# Définition des features
# -----------------------------

TARGET_COLUMN: str = "conso_5_usages_ep"

TOP_FEATURES: list[str] = [
    "etiquette_dpe",
    "surface_habitable_logement",
    "type_batiment",
    "type_generateur_n1_ecs_n1",
    "annee_construction",
    "qualite_isolation_enveloppe",
]

# (Optionnel) si tu veux les utiliser dans preprocessing
NUMERIC_FEATURES: list[str] = [
    "surface_habitable_logement",
    "annee_construction",
]

CATEGORICAL_FEATURES: list[str] = [
    "etiquette_dpe",
    "type_batiment",
    "type_generateur_n1_ecs_n1",
    "qualite_isolation_enveloppe",
]


# -----------------------------
# Fonctions utilitaires
# -----------------------------


def missing_columns(columns: Iterable[str]) -> list[str]:
    """Retourne la liste des colonnes manquantes parmi TOP_FEATURES + TARGET_COLUMN."""
    required = set(TOP_FEATURES + [TARGET_COLUMN])
    return [col for col in required if col not in columns]


def validate_model_columns(df: pd.DataFrame) -> None:
    """Vérifie que toutes les colonnes nécessaires sont présentes."""
    missing = missing_columns(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_training_columns(df: pd.DataFrame) -> None:
    """Alias pour validate_model_columns.

    Vérifie les colonnes requises pour l'entraînement.
    """
    return validate_model_columns(df)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne uniquement les features nécessaires au modèle."""
    missing = [f for f in TOP_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return df[TOP_FEATURES].copy()


def select_target(df: pd.DataFrame) -> pd.Series:
    """Retourne la colonne cible."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
    return df[TARGET_COLUMN].copy()


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Sépare X et y à partir d'un DataFrame complet."""
    return select_features(df), select_target(df)


# -----------------------------
# Création des dossiers
# -----------------------------


def create_required_directories() -> None:
    """Crée les dossiers nécessaires si absents."""
    for directory in (
        DATA_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREPROCESSORS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


create_required_directories()
