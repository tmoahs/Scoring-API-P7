# Étape 1 : Choisir une image de base
FROM python:3.11-slim

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Installer les dépendances SYSTÈME
# On installe curl pour pouvoir télécharger des fichiers
RUN apt-get update && apt-get install -y libgomp1 curl

# Étape 4 : Copier et installer les dépendances PYTHON
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Étape 5 (MODIFIÉE) : Créer les dossiers et télécharger les données ---
# On crée les dossiers dont notre application a besoin
RUN mkdir -p /app/data /app/model

# On télécharge notre dataset depuis son URL publique
# REMPLACE LE LIEN CI-DESSOUS PAR LE TIEN !
RUN curl -L -o /app/data/final_dataset.parquet "https://github.com/tmoahs/Scoring-API-P7/releases/download/v2-data/dataset_optimized.parquet"

# On copie notre code et notre modèle
COPY ./app /app/app
COPY ./model /app/model

# Étape 6 : Exposer le port
EXPOSE 8000

# Étape 7 : Définir la commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]