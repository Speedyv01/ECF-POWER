import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILENAME = "Merge_conso_meteo_soleil_090426.csv"


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Charge le dataset local.
    """

    if csv_path is None:
        csv_path = PROJECT_ROOT / DATA_FILENAME
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    return df
