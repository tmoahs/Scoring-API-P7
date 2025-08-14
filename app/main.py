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
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')  # Assure-toi que le nom du fichier est correct

    print(f"INFO: Tentative de chargement du modèle depuis: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")

except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

DATA_PATH = "data/dataset_optimized.parquet"

# --- 3. DÉFINITION DES ENDPOINTS (LES "ROUTES" DE L'API) ---

# Endpoint Racine ("/")
@app.get("/")
def read_root():
    """
    Endpoint simple pour vérifier que l'API est en ligne.
    """
    return {"message": "Bienvenue sur l'API de scoring ! L'API est en marche."}

# Endpoint de Prédiction ("/predict")
@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewLoanRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    try:
        # On passe maintenant le chemin du fichier de données
        client_data_df = prepare_data_for_prediction(
            client_id=request.SK_ID_CURR,
            new_loan_data=request.dict(),
            data_path=DATA_PATH
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ===================================================================
    # ÉTAPE DE NETTOYAGE
    # ===================================================================
    # Avant de faire la prédiction, on s'assure qu'il n'y a plus de 'None' ou 'NaN'
    # On les remplace par 0 (ou une autre valeur par défaut comme la médiane si tu préfères)
    client_data_df.fillna(0, inplace=True)

    # On vérifie qu'il n'y a plus de types 'object' qui pourraient poser problème
    for col in client_data_df.columns:
        if client_data_df[col].dtype == 'object':
            client_data_df[col] = client_data_df[col].astype(int) # ou float
    # ===================================================================

    # Étape B (TRÈS IMPORTANTE) : S'assurer que les colonnes sont dans le bon ordre
    # Le modèle s'attend à recevoir les colonnes exactement dans le même ordre que pendant l'entraînement.
    model_features = model.feature_name_
    client_data_df = client_data_df.reindex(columns=model_features, fill_value=0)

    # Étape C : Faire la prédiction
    # .predict_proba renvoie les probabilités pour les classes [0, 1]
    # On ne garde que la probabilité de la classe 1 (défaut de paiement)
    score = model.predict_proba(client_data_df)[:, 1][0]

    # Étape D : Définir la prédiction binaire (0 ou 1)
    # Tu peux ajuster ce seuil (0.5) en fonction de l'analyse que tu as faite dans tes notebooks
    prediction = 1 if score > 0.5 else 0

    # Étape E : Renvoyer la réponse finale
    # FastAPI s'occupe de la convertir au format JSON avec la structure de PredictionResponse
    return {"prediction": prediction, "score": float(score)}