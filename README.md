# ECF-POWER
Dépôt du projet fil rouge POWER formation M2i-FD1125

# Projet ML industrialisé

## Installation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Tests
pytest -v

## Qualité
ruff check .
black --check .
bandit -r src -ll
=======
