# ECF-POWER
Dépôt du projet fil rouge POWER formation M2i-FD1125
=======
# Projet ML industrialisé

## Installation
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

## Tests
`pytest -v`

## Qualité
- `ruff check .`
- `black --check .`
- `bandit -r .\api\ .\ML_CONSO_regression\src\ .\ML_DPE_regression\src\ -ll`

## Lancement
- Ouvrir powershell dans le dossier : `.venv/bin/activate`
- Lancer le backend : `uvicorn api.backend.app:app --reload`
- Ouvrir un autre powershell dans le même dossier : `.venv/bin/activate`
- Lancer le frontend : `streamlit run api/frontend/streamlit_app.py`