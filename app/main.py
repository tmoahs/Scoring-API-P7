# app/main.py (Version finale)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os

from .preprocessing import prepare_data_for_prediction
from .models import NewLoanRequest, PredictionResponse

app = FastAPI(
    title="API de Scoring de Crédit",
    version="1.0.0"
)

# --- Chargement du modèle (ne change pas) ---
try:
    model = joblib.load('model/model.pkl')
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

# On définit le chemin vers le fichier de base de données
DATA_PATH = "data/feature_store.db"  # <-- On pointe vers le fichier .db


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return JSONResponse(content={"message": "API de scoring en ligne et fonctionnelle."})


@app.post("/predict", response_model=PredictionResponse)
def predict(request: NewLoanRequest):
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

    # La suite de la logique ne change pas
    client_data_df.fillna(0, inplace=True)
    model_features = model.feature_name_
    client_data_df = client_data_df.reindex(columns=model_features, fill_value=0)

    score = model.predict_proba(client_data_df)[:, 1][0]
    prediction = 1 if score > 0.5 else 0

    return {"prediction": prediction, "score": float(score)}