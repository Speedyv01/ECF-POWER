import pytest

from src.ml_conso.data import load_data
from src.ml_conso.features import FEATURES, select_features


@pytest.fixture(scope="session")
def raw_df():
    return load_data()


@pytest.fixture(scope="session")
def selected_df(raw_df):
    return select_features(raw_df.copy())


@pytest.fixture(scope="session")
def complete_selected_df(selected_df):
    return selected_df.dropna(subset=FEATURES)
