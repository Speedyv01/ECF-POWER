from __future__ import annotations

from src.preprocessing import ProductionPreprocessor


class MLPipeline:
    """
    Pipeline simple qui encapsule uniquement le préprocesseur.
    Il sert à garder une API cohérente et propre pour train.py et predict.py.
    """

    def __init__(self):
        self.preprocessor = ProductionPreprocessor()
        self.feature_names = None

    # ---------------------------------------------------------
    # Méthodes scikit-learn like
    # ---------------------------------------------------------

    def fit(self, X, y=None):
        """
        Entraîne le préprocesseur sur X.
        """
        self.preprocessor.fit(X)
        self.feature_names = self.preprocessor.feature_names
        return self

    def transform(self, X):
        """
        Transforme X avec le préprocesseur déjà entraîné.
        """
        return self.preprocessor.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit + transform en une seule étape.
        """
        self.fit(X, y)
        return self.transform(X)

    # ---------------------------------------------------------
    # Utilitaires
    # ---------------------------------------------------------

    def get_feature_names(self):
        """
        Retourne les noms des features après preprocessing.
        """
        return self.feature_names or []
