# app/main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

# On importe les classes et fonctions de nos autres fichiers
from .preprocessing import prepare_data_for_prediction
from .models import NewLoanRequest, PredictionResponse

# --- 1. INITIALISATION DE L'API ---
# On crée l'application FastAPI.
# On peut lui donner un titre et une description qui apparaîtront dans la documentation.
app = FastAPI(
    title="API de Scoring de Crédit",
    description="Une API pour prédire la probabilité de défaut de paiement d'un client.",
    version="1.0.0"
)

# --- 2. CHARGEMENT DU MODÈLE ---
# Cette partie est exécutée une seule fois, au démarrage de l'API.
# C'est beaucoup plus efficace que de le charger à chaque requête.
try:
    model = joblib.load('model/model.pkl')
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

DATA_PATH = "data/dataset_optimized.parquet"

# --- 3. DÉFINITION DES ENDPOINTS (LES "ROUTES" DE L'API) ---

# Endpoint Racine ("/")
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    """Endpoint racine pour les vérifications de santé (health check).
    Répond explicitement aux requêtes GET et HEAD avec un statut 200 OK.
    """
    return {"message": "API de scoring en ligne et fonctionnelle."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewLoanRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    try:
        # On passe le chemin du fichier de données
        client_data_df = prepare_data_for_prediction(
            client_id=request.SK_ID_CURR,
            new_loan_data=request.dict(),
            data_path=DATA_PATH
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Logique de nettoyage et prédiction
    client_data_df.fillna(0, inplace=True)
    model_features = model.feature_name_
    client_data_df = client_data_df.reindex(columns=model_features, fill_value=0)

    score = model.predict_proba(client_data_df)[:, 1][0]
    prediction = 1 if score > 0.5 else 0

    return {"prediction": prediction, "score": float(score)}