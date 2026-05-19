import pandas as pd


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes contenant au moins une valeur manquante."""
    return df.dropna()
