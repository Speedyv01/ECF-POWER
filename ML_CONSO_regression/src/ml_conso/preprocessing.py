import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def log_transform(X):
    return np.log1p(np.clip(X, a_min=0, a_max=None))


def square_transform(X):
    return np.power(X, 2)


def build_preprocessor():
    sunshine_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    temp_log_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log_transform, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    temp_squared_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("square", FunctionTransformer(square_transform, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    humidity_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    temp_min_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("sunshine", sunshine_pipeline, ["DUREE_ENSOLEILLEMENT"]),
            ("temp_log", temp_log_pipeline, ["MOYENNE_TEMP_HORAIRES_SA_PONDEREE"]),
            ("temp_squared", temp_squared_pipeline, ["TEMP_MAX_SA"]),
            ("humidity", humidity_pipeline, ["MOYENNE_HUMIDITES_RELATIVES_HORAIRES"]),
            ("temp_min", temp_min_pipeline, ["TEMP_MIN_SOUS_ABRI"]),
        ]
    )

    return preprocessor


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare le DataFrame historique attendu par les notebooks et les tests.
    """

    from ML_CONSO_regression.src.ml_conso.features import create_target

    df = create_target(df.copy())

    numeric_columns = df.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    return df
