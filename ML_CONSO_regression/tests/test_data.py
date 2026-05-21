# tests/test_data.py
import pytest

from src.ml_conso.data import load_data


def test_load_data_file_missing(monkeypatch):
    """Doit retourner un DataFrame si le fichier n'existe pas."""
    monkeypatch.setattr("src.ml_conso.data.Path.exists", lambda self: False)

    df = load_data()

    assert not df.empty
    assert "Date" in df.columns
    assert "Conso_MWH" in df.columns
