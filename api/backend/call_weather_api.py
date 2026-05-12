from datetime import datetime

import pandas as pd
import requests


def get_forecast(zipcode: str, api_key: str):
    """Obtient les prevs météo sur 5 jours sur OpenWeatherMap"""
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"zip": f"{zipcode},fr", "appid": api_key, "units": "metric", "lang": "fr"}

    response = requests.get(url, params=params, timeout=25)
    data = response.json()

    # Si tout s'est bien passé, on obtient un code HTML 200
    if data["cod"] == "200":
        return data
    # S'il y a eu une erreur (erreur clé ou code postal), on a une autre réponse
    raise ConnectionError(f"Erreur API OpenWeatherMap : {data['message']}")


def meteo_journaliere(data: dict):
    """Extrait les infos utiles de la requête OWM et les regroupe par jour"""
    records = []
    # On crée un dictionnaire pour filtrer les infos
    for entry in data["list"]:
        records.append(
            {
                "datetime": entry["dt_txt"],
                "date": entry["dt_txt"].split(" ")[0],
                "MOYENNE_TEMP_HORAIRES_SA": entry["main"]["temp"],
                "MOYENNE_HUMIDITES_RELATIVES_HORAIRES": entry["main"]["humidity"],
                "TEMP_MIN_SOUS_ABRI": entry["main"]["temp_min"],
                "TEMP_MAX_SA": entry["main"]["temp_max"],
            }
        )
    # On transforme le dictionnaire en dataframe
    df = pd.DataFrame(records)

    # la prévision obtenue est sur 5 jours avec une ligne toutes les 3 heures
    # Agrégations journalières
    daily = (
        df.groupby("date")
        .agg(
            MOYENNE_TEMP_HORAIRES_SA=("MOYENNE_TEMP_HORAIRES_SA", "mean"),
            MOYENNE_HUMIDITES_RELATIVES_HORAIRES=(
                "MOYENNE_HUMIDITES_RELATIVES_HORAIRES",
                "mean",
            ),
            TEMP_MIN_SOUS_ABRI=("TEMP_MIN_SOUS_ABRI", "min"),
            TEMP_MAX_SA=("TEMP_MAX_SA", "max"),
        )
        .reset_index()
    )

    return daily


def get_sun(date_start: str, date_end: str):
    """Obtient la durée d'ensoleillement pour une plage de temps donnée"""
    url = "https://api.sunrisesunset.io/json"
    # On récupère les infos à l'Observatoire de Paris
    # Pour correspondre avec ce que le modèle a été entraîné
    params = {
        "lat": 48.8351933259,
        "lng": 2.33523532572,
        "date_start": date_start,
        "date_end": date_end,
        "time_format": 24,
        "timezone": "Europe/Paris",
    }

    response = requests.get(url, params=params, timeout=25)
    data = response.json()

    FMT = "%H:%M:%S"
    # On créé par un dictionnaire pour filtrer les résultats
    # et calculer la durée d'ensoleillement
    records = []
    for entry in data["results"]:
        t_start = datetime.strptime(entry["sunrise"], FMT)
        t_end = datetime.strptime(entry["sunset"], FMT)
        records.append(
            {
                "date": entry["date"],
                "sunrise": entry["sunrise"],
                "sunset": entry["sunset"],
                "DUREE_ENSOLEILLEMENT": (t_end - t_start),
            }
        )

    df = pd.DataFrame(records)

    return df


def merge_weather_sun(data: dict):
    """Fusionne les données météo et ensoleillement pour les 5 jours à venir"""
    weather = meteo_journaliere(data)
    sun = get_sun(weather["date"][0], weather["date"].iloc[-1])
    meteo_soleil = pd.merge(weather, sun, on="date", how="left")

    return meteo_soleil


# Exemple d'utilisation
if __name__ == "__main__":
    api_key = input("Entrez votre clé API : ")
    zipcode = input("Entrez un code postal : ")

    data = get_forecast(zipcode, api_key)

    if data["cod"] != "200":
        print(f"Error : {data['message']}")
        exit(1)

    meteo = merge_weather_sun(data)
    print(meteo)
