# Importations standard
from __future__ import annotations  # Pour compatibilité avec annotations futures

from pathlib import Path  # Pour les chemins de fichiers et répertoires

# Racine du projet (dossier contenant src/, data/, artifacts/, etc.)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Dossiers principaux
# Dossier pour les données brutes et traitées
DATA_DIR: Path = PROJECT_ROOT / "data"
# Dossiers pour les différents artefacts du projet
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
# Dossier pour les modèles entraînés
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
# Dossier pour les métriques d'évaluation des modèles
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
# Dossier pour les préprocesseurs sérialisés
PREPROCESSORS_DIR: Path = ARTIFACTS_DIR / "preprocessors"
# Dossier pour les prédictions générées par les modèles
PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "predictions"

# MLflow (utilisation locale par défaut)
# Configuration en mode local pour MLFLOW
MLFLOW_TRACKING_URI: str = (PROJECT_ROOT / "mlruns").resolve().as_uri()
# Nom de l'expérience MLflow
MLFLOW_EXPERIMENT_NAME: str = "prediction-energie-logement"

# Paramètres généraux d'entraînement
# Taille de l'ensemble de test (20% des données)
TEST_SIZE: float = 0.2
# Graine aléatoire pour la reproductibilité
RANDOM_STATE: int = 42

# Noms par défaut
# Nom par défaut pour le modèle entraîné
DEFAULT_MODEL_NAME: str = "DPE_model"
# Nom par défaut pour le préprocesseur sérialisé
DEFAULT_PREPROCESSOR_NAME: str = "preprocessor"
# Nom du modèle de production (bundle)
PRODUCTION_MODEL_FILENAME: str = "DPE_model_latest.pkl"


def create_required_directories() -> None:
    """Crée tous les dossiers nécessaires au projet s'ils n'existent pas."""
    for directory in (
        DATA_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREPROCESSORS_DIR,
        PREDICTIONS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_model_path(
    model_name: str = DEFAULT_MODEL_NAME, suffix: str = ".joblib"
) -> Path:
    """Retourne le chemin complet d'un modèle avec son suffixe."""
    return MODELS_DIR / f"{model_name}{suffix}"


def get_metrics_path(model_name: str) -> Path:
    """Retourne le chemin complet du fichier de métriques."""
    return METRICS_DIR / f"{model_name}_metrics.json"


def get_preprocessor_path(
    preprocessor_name: str = DEFAULT_PREPROCESSOR_NAME,
) -> Path:
    """Retourne le chemin complet du préprocesseur sérialisé."""
    return PREPROCESSORS_DIR / f"{preprocessor_name}.joblib"


def get_latest_model_info_path() -> Path:
    """Retourne le chemin du dernier meilleur modèle (fichier JSON)."""
    return MODELS_DIR / "model.latest.json"


def get_production_model_path() -> Path:
    """Retourne le chemin du bundle de production."""
    return MODELS_DIR / PRODUCTION_MODEL_FILENAME


create_required_directories()  # Création des dossiers au chargement
