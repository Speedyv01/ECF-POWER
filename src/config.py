from __future__ import annotations

from pathlib import Path

# Racine du projet (dossier contenant src/, data/, artifacts/, etc.)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Dossiers principaux
DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
PREPROCESSORS_DIR: Path = ARTIFACTS_DIR / "preprocessors"
PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "predictions"

# MLflow (utilisation locale par défaut)
MLFLOW_TRACKING_URI: str = (PROJECT_ROOT / "mlruns").resolve().as_uri()
MLFLOW_EXPERIMENT_NAME: str = "prediction-energie-logement"

# Paramètres généraux d’entraînement
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# Noms par défaut
DEFAULT_MODEL_NAME: str = "random_forest"
DEFAULT_PREPROCESSOR_NAME: str = "preprocessor"
PRODUCTION_MODEL_FILENAME: str = "modele.pkl"


def create_required_directories() -> None:
    """Crée tous les dossiers nécessaires au projet si ils n'existent pas."""
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
    """Retourne le chemin complet d'un modèle donné."""
    return MODELS_DIR / f"{model_name}{suffix}"


def get_metrics_path(model_name: str) -> Path:
    """Retourne le chemin du fichier de métriques pour un modèle donné."""
    return METRICS_DIR / f"{model_name}_metrics.json"


def get_preprocessor_path(preprocessor_name: str = DEFAULT_PREPROCESSOR_NAME) -> Path:
    """Retourne le chemin du préprocesseur sérialisé."""
    return PREPROCESSORS_DIR / f"{preprocessor_name}.joblib"


def get_latest_model_info_path() -> Path:
    """Retourne le chemin du fichier JSON décrivant le dernier meilleur modèle."""
    return MODELS_DIR / "model.latest.json"


def get_production_model_path() -> Path:
    """Retourne le chemin du bundle de production (modele.pkl)."""
    return MODELS_DIR / PRODUCTION_MODEL_FILENAME


# Création des dossiers au chargement du module
create_required_directories()
