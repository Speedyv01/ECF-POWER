import pandas as pd
import pytest
from src.ml_conso.data import load_data
from src.ml_conso.features import FEATURES, select_features


def make_sample_raw_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    day_index = pd.Series(range(len(dates)), dtype=float)

    return pd.DataFrame(
        {
            "Date": dates,
            "Conso_MWH": 1000 + day_index * 1.5 + (day_index % 7) * 8,
            "DUREE_ENSOLEILLEMENT": 4 + (day_index % 10) * 0.5,
            "MOYENNE_TEMP_HORAIRES_SA_PONDEREE": 8 + (day_index % 20) * 0.7,
            "TEMP_MAX_SA": 12 + (day_index % 18) * 0.8,
            "MOYENNE_HUMIDITES_RELATIVES_HORAIRES": 45 + (day_index % 30),
            "TEMP_MIN_SOUS_ABRI": 2 + (day_index % 15) * 0.6,
        }
    )


@pytest.fixture(scope="session")
def raw_df():
    try:
        return load_data()
    except FileNotFoundError:
        return make_sample_raw_df()


@pytest.fixture(scope="session")
def selected_df(raw_df):
    return select_features(raw_df.copy())


@pytest.fixture(scope="session")
def complete_selected_df(selected_df):
    return selected_df.dropna(subset=FEATURES)
