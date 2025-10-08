# tests/test_main.py

from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

# Créer un "client de test" pour notre API
client = TestClient(app)

# --- Test 1 : Vérifier que l'API est bien en ligne ---
def test_read_root():
    """
    Teste si l'endpoint racine ("/") répond correctement.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de scoring en ligne et fonctionnelle."}


# --- Test 2 : Vérifier une prédiction qui devrait être acceptée ---
def test_predict_accepted():
    """
    Teste l'endpoint /predict avec les données d'un client qui devrait être accepté.
    """
    client_data = {
        "SK_ID_CURR": 100025,
        "AMT_CREDIT": 1132573.5,
        "AMT_INCOME_TOTAL": 202500,
        "AMT_ANNUITY": 37561.5,
        "DAYS_BIRTH": -14815,
        "DAYS_EMPLOYED": -1652
    }

    response = client.post("/predict", json=client_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "score" in json_response
    assert json_response["prediction"] == 0  # On s'attend à ce que le prêt soit accepté


# --- Test 3 : Vérifier le comportement en cas de données manquantes ---
def test_predict_missing_data():
    """
    Teste si l'API renvoie bien une erreur 422 quand une donnée cruciale est manquante.
    """
    incomplete_data = {
        "AMT_CREDIT": 1132573.5,
        "AMT_INCOME_TOTAL": 202500,
        "AMT_ANNUITY": 37561.5,
        "DAYS_BIRTH": -14815,
        "DAYS_EMPLOYED": -1652
    }

    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422