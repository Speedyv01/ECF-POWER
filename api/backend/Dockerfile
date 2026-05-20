# Étape 1 : Construction des dépendances
FROM python:3.11-slim AS builder

WORKDIR /build

# Installation des outils nécessaires à la compilation de certaines librairies (ex: pandas/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Installation des dépendances dans un dossier isolé /install
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Étape 2 : Image finale de production (légère)
FROM python:3.11-slim

WORKDIR /app

# On récupère uniquement les dépendances installées à l'étape précédente
COPY --from=builder /install /usr/local

# On ne copie que ce qui est utile pour l'inférence
COPY ML_CONSO_regression/src/ml_conso/ ./ML_CONSO_regression/src/ml_conso/
COPY ML_DPE_regression/src/ ./ML_DPE_regression/src/
COPY api/backend/ ./api/backend/
COPY ML_DPE_regression/artifacts/models/DPE_model_latest.pkl ./ML_DPE_regression/artifacts/models/
COPY ML_CONSO_regression/artifacts/models/conso_model_latest.joblib ./ML_CONSO_regression/artifacts/models/
COPY ML_DPE_regression/artifacts/metrics/DPE_metrics_latest.json ./ML_DPE_regression/artifacts/metrics/
COPY ML_CONSO_regression/artifacts/metrics/conso_metrics_latest.json ./ML_CONSO_regression/artifacts/metrics/

# Configuration pour que Python trouve le module 'src' correctement
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
