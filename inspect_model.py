import json
import os
import sys
import types

import joblib

# Chargement des informations depuis le contrat pour garantir la cohérence
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Sécurité : On ajoute le répertoire racine au chemin de recherche de Python
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)


# --- HACK ULTIME : Création d'un module "fantôme" qui accepte tout ---
class MagicModule(types.ModuleType):
    def __getattr__(self, name):
        # Retourne une fonction vide pour tout nom demande.
        return lambda *args, **kwargs: None


# On simule la structure attendue par le modèle sans toucher à tes vrais fichiers
mock_pkg = MagicModule("src.ml_conso")
mock_pkg.__path__ = []
sys.modules["src.ml_conso"] = mock_pkg

# On simule le sous-module de preprocessing
mock_preproc = MagicModule("src.ml_conso.preprocessing")
sys.modules["src.ml_conso.preprocessing"] = mock_preproc
# --------------------------------------------------------------------

contract_path = os.path.join(base_dir, "artifacts", "conso_model_contract.json")

with open(contract_path, "r") as f:
    contract = json.load(f)

full_model_path = os.path.join(base_dir, contract["model_path"])

if os.path.exists(full_model_path):
    # Chargement - joblib trouvera désormais tout ce qu'il cherche dans nos mocks
    data = joblib.load(full_model_path)

    # Extraction du modèle (gestion du cas où c'est un dictionnaire 'bundle')
    model = data["model"] if isinstance(data, dict) and "model" in data else data

    print(f"--- Analyse du modèle : {contract['model_path']} ---")
    print(f"Algorithme utilisé : {type(model).__name__}")

    if hasattr(model, "get_params"):
        print("\nConfiguration des paramètres (Hyperparamètres) :")
        params = model.get_params()
        for key in sorted(params.keys()):
            print(f"  {key}: {params[key]}")
    else:
        print("\nL'objet chargé n'est pas un modèle Scikit-Learn standard.")
else:
    print(f"Erreur : Fichier non trouvé {full_model_path}")
