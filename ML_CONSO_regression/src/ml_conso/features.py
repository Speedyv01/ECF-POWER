import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Création de la cible evo_conso à partir de Conso_MWH écrêté.
    """

    df["Date"] = pd.to_datetime(df["Date"])

    df["jour_semaine"] = df["Date"].dt.day_name()
    df["semaine_annee"] = df["Date"].dt.isocalendar().week
    df["annee"] = df["Date"].dt.year

    df["Conso_MWH_ecrete"] = df["Conso_MWH"]

    jours_semaine = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for annee in df["annee"].unique():
        for semaine in range(1, 54):
            mask = (df["annee"] == annee) & (df["semaine_annee"] == semaine)

            df_week = df[mask]

            if df_week.empty:
                continue

            moyenne_semaine = df_week[df_week["jour_semaine"].isin(jours_semaine)][
                "Conso_MWH"
            ].mean()

            dimanche = df_week[df_week["jour_semaine"] == "Sunday"]["Conso_MWH"]

            samedi = df_week[df_week["jour_semaine"] == "Saturday"]["Conso_MWH"]

            if dimanche.empty or samedi.empty:
                continue

            delta = moyenne_semaine - dimanche.iloc[0]

            mask_jours = mask & df["jour_semaine"].isin(jours_semaine)

            df.loc[mask_jours, "Conso_MWH_ecrete"] = (
                df.loc[mask_jours, "Conso_MWH"] - delta
            )

            mask_samedi = mask & (df["jour_semaine"] == "Saturday")

            df.loc[mask_samedi, "Conso_MWH_ecrete"] = df.loc[
                mask, "Conso_MWH_ecrete"
            ].mean()

    annual_mean = df.groupby("annee")["Conso_MWH_ecrete"].transform("mean")

    df["evo_conso"] = (df["Conso_MWH_ecrete"] - annual_mean) / annual_mean
    return df


FEATURES = [
    "DUREE_ENSOLEILLEMENT",
    "MOYENNE_TEMP_HORAIRES_SA_PONDEREE",
    "TEMP_MAX_SA",
    "MOYENNE_HUMIDITES_RELATIVES_HORAIRES",
    "TEMP_MIN_SOUS_ABRI",
]

TARGET = "evo_conso"


def select_features(df):
    df = create_target(df)
    selected_columns = FEATURES + [TARGET]

    df = df[selected_columns]

    return df


def build_features(df: pd.DataFrame):
    """
    Construit X/y pour l'ancien notebook tout en restant aligne avec la cible actuelle.
    """

    df = df.copy()
    if TARGET not in df.columns:
        df = create_target(df)

    excluded_columns = {
        "Date",
        "Conso_MWH",
        "Conso_MWH_ecrete",
        "jour_semaine",
        "semaine_annee",
        "annee",
        TARGET,
        "evo_conso_scaled",
    }

    X = df.drop(columns=[column for column in excluded_columns if column in df.columns])
    X = pd.get_dummies(X, columns=["CODE_DEPARTEMENT"], drop_first=False)
    X = X.select_dtypes(include=["number", "bool"]).astype(float)
    X = X.fillna(X.median()).fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    ).fillna(0)

    scaler_target = StandardScaler()
    y_scaled_values = scaler_target.fit_transform(df[[TARGET]]).ravel()
    y = pd.Series(y_scaled_values, index=df.index, name="evo_conso_scaled")

    return X_scaled, y, scaler, scaler_target, X_scaled.columns.tolist()


def split_data(df):
    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
