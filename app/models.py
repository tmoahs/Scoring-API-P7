# app/models.py
from pydantic import BaseModel
from typing import Optional

# --------------------------------------------------------------------
# 1. MODÈLE POUR LA REQUÊTE (LES DONNÉES EN ENTRÉE)
# --------------------------------------------------------------------
class NewLoanRequest(BaseModel):
    """
    Définit la structure des données pour une nouvelle demande de prêt.
    Ce sont les informations que l'utilisateur de l'API doit fournir.
    """
    # L'identifiant unique du client est obligatoire
    SK_ID_CURR: int

    # On liste ici les autres champs qui peuvent être fournis
    # lors d'une nouvelle demande. On se base sur les colonnes
    # de 'application_train.csv' qui sont pertinentes.
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: Optional[float] = None # 'Optional' si le champ peut être manquant
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    CNT_CHILDREN: Optional[int] = None

    # Cette partie est un bonus : elle ajoute un exemple directement
    # dans la documentation de l'API pour que ce soit plus facile à tester.
    class Config:
        schema_extra = {
            "example": {
                "SK_ID_CURR": 100002,
                "AMT_CREDIT": 406597.5,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_ANNUITY": 24700.5,
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637
            }
        }

# --------------------------------------------------------------------
# 2. MODÈLE POUR LA RÉPONSE (LES DONNÉES EN SORTIE)
# --------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """
    Définit la structure de la réponse de l'API.
    """
    prediction: int      # 0 pour "accepté", 1 pour "refusé"
    score: float         # La probabilité de défaut (entre 0 et 1)