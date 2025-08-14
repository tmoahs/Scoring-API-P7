# app/main.py (Version finale avec SHAP)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import shap  # <-- 1. Importer SHAP

from .preprocessing import prepare_data_for_prediction
from .models import NewLoanRequest, PredictionResponse

app = FastAPI(
    title="API de Scoring de Crédit",
    version="1.0.0"
)

# --- Chargement du modèle et de l'explainer SHAP ---
try:
    model = joblib.load('model/model.pkl')
    print("✅ Modèle chargé avec succès.")

    # 2. Créer un explainer SHAP au démarrage
    # Il est créé une seule fois pour être réutilisé, c'est plus efficace.
    explainer = shap.TreeExplainer(model)
    print("✅ Explainer SHAP créé avec succès.")

except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle ou de l'explainer : {e}")
    model = None
    explainer = None

# On définit le chemin vers le fichier de base de données
DATA_PATH = "data/feature_store.db"


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return JSONResponse(content={"message": "API de scoring en ligne et fonctionnelle."})


@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewLoanRequest):
    # ... (le code de cette fonction ne change pas)
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    try:
        client_data_df = prepare_data_for_prediction(
            client_id=request.SK_ID_CURR,
            new_loan_data=request.dict(),
            db_path=DATA_PATH
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    client_data_df.fillna(0, inplace=True)
    model_features = model.feature_name_
    client_data_df = client_data_df.reindex(columns=model_features, fill_value=0)

    score = model.predict_proba(client_data_df)[:, 1][0]
    prediction = 1 if score > 0.5 else 0

    return {"prediction": prediction, "score": float(score)}


# --- 3. NOUVEL ENDPOINT POUR LES EXPLICATIONS SHAP ---
@app.get("/shap_explanation/{client_id}")
def get_shap_explanation(client_id: int):
    """
    Fournit les données nécessaires pour une explication SHAP pour un client donné.
    """
    if explainer is None or model is None:
        raise HTTPException(status_code=503, detail="Explainer SHAP non disponible.")

    try:
        # On récupère les données du client (sans les infos de la nouvelle demande)
        client_data_df = prepare_data_for_prediction(
            client_id=client_id,
            new_loan_data={"SK_ID_CURR": client_id},  # On passe un dict minimal
            db_path=DATA_PATH
        )
        client_data_df.fillna(0, inplace=True)
        model_features = model.feature_name_
        client_data_df = client_data_df.reindex(columns=model_features, fill_value=0)

        # Calculer les valeurs SHAP pour ce client
        shap_values = explainer.shap_values(client_data_df)

        # Formater la réponse pour qu'elle soit facile à utiliser
        response_data = {
            "base_value": explainer.expected_value[1],  # Valeur de base pour la classe 1 (défaut)
            "shap_values": shap_values[1][0].tolist(),  # Valeurs SHAP pour la classe 1
            "feature_names": client_data_df.columns.tolist(),
            "feature_values": client_data_df.iloc[0].tolist()
        }
        return response_data

    except ValueError as e:  # Le client n'existe pas
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul SHAP : {e}")