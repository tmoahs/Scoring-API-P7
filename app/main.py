# app/main.py (Version finale avec SHAP)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import shap
import threading

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

# On définit le chemin vers le fichier de log des prédictions
PREDICTIONS_LOG_PATH = "data/predictions_log.csv"

# Un verrou pour éviter les problèmes d'écriture simultanée sur le fichier
file_lock = threading.Lock()


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

    # On fait une copie des données avant la prédiction pour les sauvegarder
    data_to_log = client_data_df.copy()

    client_data_df.fillna(0, inplace=True)
    model_features = model.feature_name_
    client_data_df = client_data_df.reindex(columns=model_features, fill_value=0)

    score = model.predict_proba(client_data_df)[:, 1][0]
    prediction = 1 if score > 0.5 else 0

    # On ajoute le score et la prédiction au DataFrame à sauvegarder
    data_to_log['SCORE'] = score
    data_to_log['PREDICTION'] = prediction

    # On sauvegarde dans le fichier CSV en utilisant un verrou
    with file_lock:
        data_to_log.to_csv(
            PREDICTIONS_LOG_PATH,
            mode='a',  # 'a' pour "append" (ajouter à la fin)
            header=not os.path.exists(PREDICTIONS_LOG_PATH),  # N'écrit l'en-tête que si le fichier n'existe pas
            index=False
        )

    return {"prediction": prediction, "score": float(score)}


# --- 3. NOUVEL ENDPOINT POUR LES EXPLICATIONS SHAP ---
@app.get("/shap_explanation/{client_id}")
def get_shap_explanation(client_id: int):
    """
    Fournit les données nécessaires pour une explication SHAP pour un client donné.
    Cette version est robuste aux changements de format de la librairie SHAP.
    """
    if explainer is None or model is None:
        raise HTTPException(status_code=503, detail="Explainer SHAP non disponible.")

    try:
        # On récupère les données du client
        client_data_df = prepare_data_for_prediction(
            client_id=client_id,
            new_loan_data={"SK_ID_CURR": client_id},
            db_path=DATA_PATH
        )
        client_data_df.fillna(0, inplace=True)
        model_features = model.feature_name_
        client_data_df = client_data_df.reindex(columns=model_features, fill_value=0)

        # Calculer les valeurs SHAP pour ce client
        shap_values_output = explainer.shap_values(client_data_df)

        # --- Logique robuste pour gérer les différentes versions de SHAP ---
        # Pour la valeur de base (expected_value)
        if isinstance(explainer.expected_value, list):
            base_value = explainer.expected_value[1]  # On prend la valeur pour la classe 1 (défaut)
        else:
            base_value = explainer.expected_value

        # Pour les valeurs SHAP
        if isinstance(shap_values_output, list):
            shap_values_for_prediction = shap_values_output[1][0]  # On prend les valeurs pour la classe 1
        else:
            shap_values_for_prediction = shap_values_output[0]

        # Formater la réponse pour qu'elle soit facile à utiliser
        response_data = {
            "base_value": base_value,
            "shap_values": shap_values_for_prediction.tolist(),
            "feature_names": client_data_df.columns.tolist(),
            "feature_values": client_data_df.iloc[0].tolist()
        }
        return response_data

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Cette ligne aidera à voir l'erreur précise dans les logs de Render si elle persiste
        import traceback
        print(f"Erreur détaillée dans get_shap_explanation: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul SHAP : {e}")