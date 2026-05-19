import joblib

from ML_DPE_regression.src.config import get_production_model_path


class ModelLoader:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = None

    def load_latest_model(self, model_path=None):
        model_path = model_path or get_production_model_path()
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.metadata = bundle["metadata"]
        return self

    def is_loaded(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def get_model(self):
        if not self.is_loaded():
            raise ValueError("Model not loaded. Call load_latest_model() first.")
        return self.model

    def get_preprocessor(self):
        if not self.is_loaded():
            raise ValueError("Preprocessor not loaded. Call load_latest_model() first.")
        return self.preprocessor

    def get_metadata(self):
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_latest_model() first.")
        return self.metadata
