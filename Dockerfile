# =================================================================
# RECETTE DE CONSTRUCTION DE NOTRE CONTENEUR D'API
# =================================================================

# --- Étape 1: Le Point de Départ ---
# On choisit notre fondation. Ici, une image officielle et légère
# qui contient déjà Python dans sa version 3.11.
FROM python:3.11-slim

# --- Étape 2: L'Espace de Travail ---
# On crée un dossier nommé "/app" à l'intérieur du conteneur.
# Ce sera le dossier principal de notre application dans la "boîte".
WORKDIR /app

# On met à jour la liste des paquets et on installe la librairie manquante pour LightGBM
RUN apt-get update && apt-get install -y libgomp1

# --- Étape 3: Installation des Dépendances ---
# On copie d'abord SEULEMENT le fichier requirements.txt.
# Pourquoi ? Docker a un système de cache intelligent. Si on ne change
# que notre code (app/) sans changer les librairies, Docker n'aura
# pas besoin de réinstaller toutes les dépendances, ce qui rendra
# les constructions futures beaucoup plus rapides.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Étape 4: Ajout de Notre Application ---
# Maintenant que les fondations sont posées, on copie notre code
# et nos données dans le conteneur.
COPY ./app /app/app
COPY ./data /app/data
COPY ./model /app/model

# --- Étape 5: Ouvrir la Porte ---
# On indique que notre application, à l'intérieur du conteneur,
# va utiliser le port 8000 pour communiquer avec l'extérieur.
EXPOSE 8000

# --- Étape 6: La Commande de Lancement ---
# C'est LA commande qui sera exécutée automatiquement quand on démarrera le conteneur.
# Elle lance le serveur Uvicorn, le rend accessible depuis l'extérieur du conteneur (--host 0.0.0.0)
# et lui dit d'utiliser le port 8000.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]