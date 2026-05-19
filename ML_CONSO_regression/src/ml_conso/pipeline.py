import json
import shutil
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.ml_conso.data import load_data
from src.ml_conso.evaluate import evaluate_model
from src.ml_conso.features import select_features, split_data
from src.ml_conso.preprocessing import build_preprocessor
from src.ml_conso.train import build_model


def build_pipeline():

    preprocessor = build_preprocessor()

    model = build_model()

    pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", model)])

    return pipeline


def save_model_version(model, metrics: dict, version: str, artifact_dir="artifacts"):
    artifact_dir = Path(artifact_dir)
    models_dir = artifact_dir / "models"
    metrics_dir = artifact_dir / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"conso_model_{version}.joblib"
    metrics_path = metrics_dir / f"conso_metrics_{version}.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    return {"model": model_path, "metrics": metrics_path}


def activate_model_version(version: str, artifact_dir="artifacts"):
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / "models" / f"conso_model_{version}.joblib"
    metrics_path = artifact_dir / "metrics" / f"conso_metrics_{version}.json"
    latest_model_path = artifact_dir / "models" / "conso_model_latest.joblib"
    latest_metrics_path = artifact_dir / "metrics" / "conso_metrics_latest.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable : {model_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metriques introuvables : {metrics_path}")

    latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_path, latest_model_path)
    shutil.copy2(metrics_path, latest_metrics_path)

    return {"model": latest_model_path, "metrics": latest_metrics_path}


def load_latest_model(artifact_dir="artifacts"):
    model_path = Path(artifact_dir) / "models" / "conso_model_latest.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modele latest introuvable : {model_path}")

    return joblib.load(model_path)


def export_model_contract(feature_names, artifact_dir="artifacts"):
    artifact_dir = Path(artifact_dir)
    contract_path = artifact_dir / "model_contract.json"

    contract = {
        "model_path": "artifacts/models/conso_model_latest.joblib",
        "metrics_path": "artifacts/metrics/conso_metrics_latest.json",
        "features": list(feature_names),
        "target": "evo_conso",
        "format": "joblib",
        "usage": """Load in the backend with joblib,
        expose predictions through an API for the frontend.""",
    }

    with contract_path.open("w", encoding="utf-8") as file:
        json.dump(contract, file, indent=4)

    return contract_path


def run_pipeline():
    df = load_data()
    df = select_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    evaluation = evaluate_model(pipeline, X_test, y_test)

    return {
        "best_model": pipeline,
        "evaluation": evaluation,
        "feature_names": X_train.columns.tolist(),
    }
