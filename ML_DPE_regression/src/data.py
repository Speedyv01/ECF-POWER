import pandas as pd

from src.config import DATA_DIR


def load_raw_data(filename: str = "dpe_processed_03032026.csv") -> pd.DataFrame:
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pd.read_csv(
        file_path,
        sep=";",
        quotechar='"',
        decimal=",",
        encoding="utf-8",
        low_memory=False,
    )
