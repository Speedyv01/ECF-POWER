FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances minimales
COPY api/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie uniquement du dossier frontend
COPY api/frontend/ ./api/frontend/

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "api/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
