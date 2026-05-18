import pandas as pd
import pytest
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import TARGET_COLUMN  # noqa: E402


@pytest.fixture
def workspace_tmp_path(request):
    root = PROJECT_ROOT / ".pytest_tmp"
    path = root / request.node.name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def sample_df():
    rows = [
        {
            "etiquette_dpe": "A",
            "surface_habitable_logement": 45.0,
            "type_batiment": "appartement",
            "type_generateur_n1_ecs_n1": "Ballon électrique",
            "annee_construction": 1998,
            "qualite_isolation_enveloppe": "bonne",
            "conso_5_usages_par_m2_ep": 120.0,
            "apport_solaire_saison_chauffe": 12.0,
            "deperditions_enveloppe": 80.0,
            "classe_inertie_batiment": "moyenne",
            TARGET_COLUMN: 5400.0,
        },
        {
            "etiquette_dpe": "B",
            "surface_habitable_logement": 62.0,
            "type_batiment": "maison",
            "type_generateur_n1_ecs_n1": "Pompe à chaleur",
            "annee_construction": 2005,
            "qualite_isolation_enveloppe": "tres bonne",
            "conso_5_usages_par_m2_ep": 95.0,
            "apport_solaire_saison_chauffe": 15.0,
            "deperditions_enveloppe": 75.0,
            "classe_inertie_batiment": "lourde",
            TARGET_COLUMN: 5890.0,
        },
        {
            "etiquette_dpe": "C",
            "surface_habitable_logement": 80.0,
            "type_batiment": "maison",
            "type_generateur_n1_ecs_n1": "Chaudière gaz",
            "annee_construction": 1980,
            "qualite_isolation_enveloppe": "moyenne",
            "conso_5_usages_par_m2_ep": 150.0,
            "apport_solaire_saison_chauffe": 8.0,
            "deperditions_enveloppe": 110.0,
            "classe_inertie_batiment": "legere",
            TARGET_COLUMN: 12000.0,
        },
        {
            "etiquette_dpe": "D",
            "surface_habitable_logement": 55.0,
            "type_batiment": "appartement",
            "type_generateur_n1_ecs_n1": "Ballon électrique",
            "annee_construction": 1975,
            "qualite_isolation_enveloppe": "insuffisante",
            "conso_5_usages_par_m2_ep": 180.0,
            "apport_solaire_saison_chauffe": 6.0,
            "deperditions_enveloppe": 140.0,
            "classe_inertie_batiment": "moyenne",
            TARGET_COLUMN: 9900.0,
        },
        {
            "etiquette_dpe": "E",
            "surface_habitable_logement": 70.0,
            "type_batiment": "maison",
            "type_generateur_n1_ecs_n1": "Chaudière fioul",
            "annee_construction": 1965,
            "qualite_isolation_enveloppe": "moyenne",
            "conso_5_usages_par_m2_ep": 210.0,
            "apport_solaire_saison_chauffe": 5.0,
            "deperditions_enveloppe": 165.0,
            "classe_inertie_batiment": "lourde",
            TARGET_COLUMN: 14700.0,
        },
        {
            "etiquette_dpe": "F",
            "surface_habitable_logement": 90.0,
            "type_batiment": "maison",
            "type_generateur_n1_ecs_n1": "Chaudière gaz",
            "annee_construction": 1950,
            "qualite_isolation_enveloppe": "insuffisante",
            "conso_5_usages_par_m2_ep": 240.0,
            "apport_solaire_saison_chauffe": 4.0,
            "deperditions_enveloppe": 190.0,
            "classe_inertie_batiment": "tres lourde",
            TARGET_COLUMN: 21600.0,
        },
    ]
    return pd.DataFrame(rows)
