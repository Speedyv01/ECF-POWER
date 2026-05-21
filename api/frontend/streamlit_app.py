import logging
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PREDICT_DPE_URL = f"{BACKEND_URL}/predict_DPE"
PREDICT_CONSO_URL = f"{BACKEND_URL}/predict_conso"

logger.info("Demarrage de Streamlit. BACKEND_URL=%s", BACKEND_URL)

if "prediction" not in st.session_state:
    st.session_state.prediction = 0

# Premier bloc: prédiction conso selon caractéristiques logement (modèle DPE)
st.title(" :house: :green[Prédiction de la consommation énergétique d'un logement]")
st.subheader("Saisissez les caractéristiques du logement pour obtenir une estimation.")


etiquette_dpe = st.selectbox(
    "🏷️ Étiquette DPE", ["C", "A", "B", "D", "E", "F", "G"]
)  # C en premier par défaut
surface_habitable = st.slider(
    "📏 Surface en m²", min_value=0, max_value=1000, step=1, value=50
)
type_batiment = st.selectbox("🏢Type de logement", ["maison", "appartement"])
type_generateur_n1_ecs_n1 = st.selectbox(
    "🚿 Type de générateur d'eau chaude sanitaire",
    [
        "Système de production d'ecs non électrique",
        "Pompe à chaleur",
        "Chauffe-eau thermodynamique",
        "Ballon électrique",
        "Chauffe-eau électrique",
        "Chaudière électrique",
    ],
)
annee_construction = st.slider(
    "🗓️Année de construction",
    min_value=1800,
    max_value=datetime.now().year,
    value=1950,
    step=1,
)
qualite_isolation_enveloppe = st.selectbox(
    "Qualité de l'isolation du logement",
    ["moyenne", "insuffisante", "bonne", "très bonne"],
)

# Premier bouton: appelle predict_DPE
if st.button("Calculer l'estimation", type="primary"):
    payload = {  # Il faut que le nom des variables match ceux du modèle
        "etiquette_dpe": etiquette_dpe,
        "surface_habitable_logement": surface_habitable,
        "type_batiment": type_batiment,
        "type_generateur_n1_ecs_n1": type_generateur_n1_ecs_n1,
        "annee_construction": annee_construction,
        "qualite_isolation_enveloppe": qualite_isolation_enveloppe,
    }

    try:
        logger.info("Envoi de la requete HTTP de prediction vers %s", PREDICT_DPE_URL)
        response = requests.post(PREDICT_DPE_URL, json=payload, timeout=5)
        response.raise_for_status()
        prediction = response.json()["prediction"]
        # Stocker la prédiction dans session_state
        st.session_state.prediction = prediction
        st.success(f"### Conso annuelle estimée : {prediction:.2f} kWh")
    except requests.exceptions.ConnectionError:
        logger.error("Erreur : Impossible de contacter le serveur backend (FastAPI).")
        st.error("Erreur : Impossible de contacter le serveur backend (FastAPI).")
    except Exception as e:
        logger.error("Une erreur est survenue : %s", e)
        st.error(f"Une erreur est survenue : {e}")

st.space("small")
st.title(" ⛅ :green[Évolution de la consommation]")
st.subheader(
    "Saisissez vos identifiants pour obtenir l'évolution de votre consommation"
)

conso_annuelle = st.number_input(
    label="Entrez votre conso annuelle (estimée ou réelle) en kWh",
    min_value=0.00,
    max_value=100000.00,
    value=float(st.session_state.prediction),
    step=0.10,
    format="%0.2f",
)
zipcode = st.text_input("📫 Code postal", value="75014")  # CP Observatoire de Paris
api_key = st.text_input("🔐 Entrez votre clé API OpenWeatherMap", type="password")

# Deuxième bouton: appelle predict_conso
if st.button("Obtenir l'évolution", type="primary"):
    if not api_key:  # Pas de clé, on affiche une erreur
        st.error("Veuillez entrer votre clé API OpenWeatherMap.")
    else:
        # Si pas de prediction de conso faite, on affiche la météo
        if conso_annuelle == 0:
            conso_jour = None
        else:
            conso_jour = conso_annuelle / 365

        payload = {"zipcode": zipcode, "api_key": api_key, "conso_jour": conso_jour}

        try:
            logger.info(
                "Envoi de la requete HTTP de prediction vers %s", PREDICT_CONSO_URL
            )
            response = requests.post(PREDICT_CONSO_URL, json=payload, timeout=15)
            response.raise_for_status()
            prediction = response.json()["predictions"]
            ville = response.json()["city_name"]
            st.write(f"Prévisions météo pour {ville} :")
            st.dataframe(
                prediction,
                hide_index=True,
                column_order=(
                    "Date",
                    "Lever 🌅",
                    "Coucher 🌇",
                    "T° min",
                    "T° max",
                    "T° moyenne",
                    "Humidité moyenne",
                    "Conso",
                ),
            )
            df = pd.DataFrame(prediction)
            st.success("### Conso à venir dans les 6 prochains jours :")
            st.success(f"{df['Conso'].sum():.2f} kWh")
        except requests.exceptions.ConnectionError:
            logger.error(
                "Erreur : Impossible de contacter le serveur backend (FastAPI)."
            )
            st.error("Erreur : Impossible de contacter le serveur backend (FastAPI).")
        except Exception as e:
            logger.error("Une erreur est survenue : %s", e)
            st.error(f"Une erreur est survenue : {e}")
