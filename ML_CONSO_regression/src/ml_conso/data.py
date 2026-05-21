from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILENAME = "Merge_conso_meteo_soleil_090426.csv"


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "Date": rng,
            "Conso_MWH": np.random.uniform(1000, 5000, size=n),
            "DUREE_ENSOLEILLEMENT": np.random.uniform(0, 10, size=n),
            "MOYENNE_TEMP_HORAIRES_SA_PONDEREE": np.random.uniform(0, 30, size=n),
            "TEMP_MAX_SA": np.random.uniform(5, 35, size=n),
            "MOYENNE_HUMIDITES_RELATIVES_HORAIRES": np.random.uniform(30, 90, size=n),
            "TEMP_MIN_SOUS_ABRI": np.random.uniform(-5, 20, size=n),
            "CODE_DEPARTEMENT": np.random.choice([1, 2, 3], size=n),
        }
    )
    return df


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """
    Charge le dataset local.

    Si le fichier est introuvable, retourne un jeu de données synthétique de secours.
    """

    if csv_path is None:
        csv_path = PROJECT_ROOT / DATA_FILENAME
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        message = (
            f"WARN: data file not found at {csv_path!s} "
            "— creating synthetic dataset for dev."
        )
        print(message)
        return _make_synthetic_df()

    df = pd.read_csv(csv_path)

    return df
