from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from .src.ml_conso.data import load_data
from .src.ml_conso.evaluate import evaluate_model
from .src.ml_conso.features import select_features, split_data
from .src.ml_conso.pipeline import (
    activate_model_version,
    build_pipeline,
    export_model_contract,
    save_model_version,
)
from .src.ml_conso.train import log_experiment

# Fonction d'aide pour afficher des en-têtes de section de manière cohérente dans la console, ce qui améliore la lisibilité des logs lors de l'exécution du pipeline.
def log_step(message):
    """Helper to print section headers consistently."""
    print("=" * 60)
    print(message)
    print("=" * 60)

# Fonction principale pour exécuter le pipeline complet, qui inclut le chargement des données, la sélection des features, la division des données en ensembles d'entraînement et de test, la construction du pipeline, l'entraînement du modèle, l'évaluation des performances et le retour des résultats.
def main():
    log_step("CHARGEMENT DES DONNEES")

    df = load_data()
    print(df.shape)

    log_step("SELECTION DES FEATURES")

    df = select_features(df)
    print(df.columns.tolist())

    log_step("SPLIT TRAIN / TEST")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train : {X_train.shape}")
    print(f"Test : {X_test.shape}")

    log_step("PIPELINE")

    pipeline = build_pipeline()

    mlflow.set_experiment("electricity_forecasting") #mlflow.set_experiment pour définir le nom de l'expérience MLflow sous laquelle les résultats seront enregistrés, ce qui permet d'organiser et de suivre les différentes expériences de manière structurée.

    with mlflow.start_run(): #mlflow.start_run pour démarrer une nouvelle exécution MLflow, ce qui permet d'enregistrer les paramètres, les métriques et les artefacts associés à cette exécution de manière isolée et traçable.
        log_step("ENTRAINEMENT")

        pipeline.fit(X_train, y_train)

        log_step("EVALUATION")

        metrics = evaluate_model(pipeline, X_test, y_test)

        print(metrics)

        log_experiment(metrics)

        log_step("SAUVEGARDE")

        # Remonte de un niveau : de main.py -> ML_CONSO_regression
        artifact_dir = Path(__file__).resolve().parent / "artifacts"
        artifact_dir.mkdir(exist_ok=True)

        save_model_version(pipeline, metrics, "current", artifact_dir)

        activate_model_version("current", artifact_dir)

        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model")

        joblib.dump(X_train.columns.tolist(), artifact_dir / "feature_columns.pkl")

        export_model_contract(X_train.columns.tolist(), artifact_dir)

        print("Model saved.")

    log_step("FIN")


if __name__ == "__main__":
    main()
