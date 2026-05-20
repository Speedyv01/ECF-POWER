# tests/test_data.py
import pytest

from src.ml_conso.data import load_data


def test_load_data_file_missing(monkeypatch):
    """Doit lever FileNotFoundError si le fichier n'existe pas."""
    monkeypatch.setattr("src.ml_conso.data.Path.exists", lambda self: False)
    with pytest.raises(FileNotFoundError):
        load_data()
