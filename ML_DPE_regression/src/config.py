# Importations standard
from __future__ import annotations # Importation pour compatibilité avec les annotations de type futures

from pathlib import Path # Importation des chemins de fichiers et des répertoires

# Racine du projet (dossier contenant src/, data/, artifacts/, etc.)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1] 

# Dossiers principaux
DATA_DIR: Path = PROJECT_ROOT / "data" # Dossier pour les données brutes et traitées
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts" # Dossiers pour les différents artefacts du projet (modèles, métriques, préprocesseurs, etc.)
MODELS_DIR: Path = ARTIFACTS_DIR / "models" # Dossier pour les modèles entraînés
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics" # Dossier pour les métriques d'évaluation des modèles
PREPROCESSORS_DIR: Path = ARTIFACTS_DIR / "preprocessors" # Dossier pour les préprocesseurs sérialisés
PREDICTIONS_DIR: Path = ARTIFACTS_DIR / "predictions" # Dossier pour les prédictions générées par les modèles

# MLflow (utilisation locale par défaut)
MLFLOW_TRACKING_URI: str = (PROJECT_ROOT / "mlruns").resolve().as_uri() # Configuration en mode local pour MLFLOW
MLFLOW_EXPERIMENT_NAME: str = "prediction-energie-logement" # Nom de l'expérience MLflow pour le suivi des expériences d'entraînement et d'évaluation des modèles

# Paramètres généraux d’entraînement
TEST_SIZE: float = 0.2 #Taille de l'ensemble de test (20% des données)
RANDOM_STATE: int = 42 # Graine aléatoire pour la reproductibilité des résultats lors de la division des données et de l'entraînement des modèles

# Noms par défaut
DEFAULT_MODEL_NAME: str = "DPE_model" # Nom par défaut pour le modèle entraîné
DEFAULT_PREPROCESSOR_NAME: str = "preprocessor" # Nom par défaut pour le préprocesseur sérialisé
PRODUCTION_MODEL_FILENAME: str = "DPE_model_latest.pkl" # Nom du modèle de production (bundle)


def create_required_directories() -> None: # Garantie que tous les dossiers nécessaires au projet existent, sinon ils sont créés.
    for directory in (
        DATA_DIR, # dossier pour les données brutes et traitées
        ARTIFACTS_DIR, # dossier pour les différents artefacts du projet
        MODELS_DIR, # dossier pour les modèles entraînés
        METRICS_DIR, # dossier pour les métriques
        PREPROCESSORS_DIR, # dossier pour les préprocesseurs sérialisés
        PREDICTIONS_DIR, # dossier pour les prédictions générées par les modèles
    ):
        directory.mkdir(parents=True, exist_ok=True) # Crée le dossier s'il n'existe pas déjà, en créant également les dossiers parents si nécessaire.


def get_model_path( # chemin complet d'un modèle donné, avec un suffixe par défaut ".joblib" pour la sérialisation.
    model_name: str = DEFAULT_MODEL_NAME, suffix: str = ".joblib"
) -> Path:
    return MODELS_DIR / f"{model_name}{suffix}"


def get_metrics_path(model_name: str) -> Path: #chemin complet du fichier de métriques pour un modèle donné, avec un suffixe par défaut ".json" pour la sérialisation.
    return METRICS_DIR / f"{model_name}_metrics.json"


def get_preprocessor_path(preprocessor_name: str = DEFAULT_PREPROCESSOR_NAME) -> Path: # chemin complet du préprocesseur sérialisé pour un nom de préprocesseur donné, avec un suffixe par défaut ".joblib" pour la sérialisation.
    return PREPROCESSORS_DIR / f"{preprocessor_name}.joblib"


def get_latest_model_info_path() -> Path: # chemin complet du fichier JSON décrivant le dernier meilleur modèle, avec un suffixe par défaut ".json" pour la sérialisation.
    return MODELS_DIR / "model.latest.json"


def get_production_model_path() -> Path: #Retourne le chemin du bundle de production (modele.pkl)    
    return MODELS_DIR / PRODUCTION_MODEL_FILENAME


create_required_directories() # Création des dossiers au chargement du module
