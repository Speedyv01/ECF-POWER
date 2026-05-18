from __future__ import annotations

import argparse
import pandas as pd

from src.train import train
from src.predict import predict
from src.evaluate import evaluate

DEFAULT_DATASET = "data/dpe_processed_03032026.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline ML Fil Rouge")

    parser.add_argument(
        "mode",
        type=str,
        nargs="?",
        choices=["train", "predict", "evaluate"],
        help="Mode d'exécution : train, predict ou evaluate",
        default=None,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Chemin vers un fichier CSV",
    )

    return parser.parse_args()


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8",
        engine="python",
    )


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------
    # MODE AUTO (RUN VS CODE)
    # ---------------------------------------------------------
    if args.mode is None:
        print("▶️ Lancement automatique (RUN VS Code détecté)")
        print(f"📄 Dataset par défaut : {DEFAULT_DATASET}")

        df = read_csv(DEFAULT_DATASET)
        _, _, metrics = train(df)

        best_metrics = metrics["best_metrics"]
        print("\n🎉 Entraînement terminé.")
        print(f"🏆 Meilleur modèle : {metrics['best_model']}")
        print(f"   R²   : {best_metrics['r2']:.4f}")
        print(f"   MAE  : {best_metrics['mae']:.4f}")
        print(f"   RMSE : {best_metrics['rmse']:.4f}")
        return

    # ---------------------------------------------------------
    # MODE TRAIN
    # ---------------------------------------------------------
    if args.mode == "train":
        if not args.input:
            print("❌ Erreur : --input est requis en mode train")
            return

        print("🚀 Lancement de l'entraînement du modèle...")
        df = read_csv(args.input)
        _, _, metrics = train(df)

        best_metrics = metrics["best_metrics"]
        print("\n🎉 Entraînement terminé.")
        print(f"🏆 Meilleur modèle : {metrics['best_model']}")
        print(f"   R²   : {best_metrics['r2']:.4f}")
        print(f"   MAE  : {best_metrics['mae']:.4f}")
        print(f"   RMSE : {best_metrics['rmse']:.4f}")
        return

    # ---------------------------------------------------------
    # MODE PREDICT
    # ---------------------------------------------------------
    if args.mode == "predict":
        if not args.input:
            print("❌ Erreur : --input est requis en mode predict")
            return

        print("🔮 Lancement de la prédiction...")
        df = read_csv(args.input)
        y_pred, metadata = predict(df)

        print("\n=== RÉSULTATS ===")
        print("Prédictions :", y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred)
        print("Modèle utilisé :", metadata.get("best_model"))
        return

    # ---------------------------------------------------------
    # MODE EVALUATE
    # ---------------------------------------------------------
    if args.mode == "evaluate":
        if not args.input:
            print("❌ Erreur : --input est requis en mode evaluate")
            return

        print("📊 Évaluation du modèle...")
        df = read_csv(args.input)
        metrics = evaluate(df)

        print("\n🎉 Évaluation terminée.")
        print(f"   R²   : {metrics['r2']:.4f}")
        print(f"   MAE  : {metrics['mae']:.4f}")
        print(f"   RMSE : {metrics['rmse']:.4f}")
        return


if __name__ == "__main__":
    main()
