# app/preprocessing.py
import pandas as pd
import numpy as np
import os # <-- Ajoute cet import

# --- ÉTAPE 1: CHARGEMENT DU MAGASIN DE FEATURES ---

# On construit un chemin d'accès "intelligent" et robuste
# __file__ est une variable spéciale qui contient le chemin du fichier actuel (preprocessing.py)
# os.path.dirname() prend le dossier parent de ce chemin
# On remonte de deux niveaux pour arriver à la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# On combine le chemin de la racine avec le dossier 'data' et le nom du fichier
DATA_PATH = os.path.join(BASE_DIR, 'data', 'final_dataset.parquet')

try:
    print(f"INFO: Tentative de chargement du magasin de features depuis: {DATA_PATH}")
    FEATURE_STORE = pd.read_parquet(DATA_PATH)
    print("✅ Magasin de features chargé avec succès.")
    # On met l'ID client en index pour des recherches ultra-rapides (lookup)
    FEATURE_STORE.set_index('SK_ID_CURR', inplace=True)
except FileNotFoundError:
    print(f"❌ ERREUR: Fichier introuvable à l'adresse: {DATA_PATH}")
    print("❌ ERREUR: Assurez-vous que le fichier 'final_dataset.parquet' se trouve bien dans le dossier 'data/' à la racine du projet.")
    FEATURE_STORE = None


# --- ÉTAPE 2: LA FONCTION DE PRÉPARATION POUR L'API ---
def prepare_data_for_prediction(client_id: int, new_loan_data: dict) -> pd.DataFrame:
    """
    Prépare la ligne de données finale pour un client donné.

    Args:
        client_id (int): Le SK_ID_CURR du client.
        new_loan_data (dict): Un dictionnaire avec les données de la nouvelle demande.

    Returns:
        pd.DataFrame: Un DataFrame avec une seule ligne, prêt pour le modèle.
    """
    epsilon = 1e-6

    if FEATURE_STORE is None:
        raise RuntimeError("Le magasin de features n'est pas disponible.")

    # 1. Récupérer les features pré-calculées pour le client
    try:
        precalculated_features = FEATURE_STORE.loc[[client_id]].copy()
    except KeyError:
        # Si le client n'a aucun historique, on ne peut pas prédire
        raise ValueError(f"Client avec SK_ID_CURR {client_id} non trouvé dans l'historique.")

    # 2. Créer un DataFrame pour les nouvelles données
    new_data_df = pd.DataFrame([new_loan_data])

    # 3. Mettre à jour les features pré-calculées avec les nouvelles données
    # La nouvelle demande de crédit met à jour les informations de base
    for col in new_data_df.columns:
        if col in precalculated_features.columns:
            precalculated_features[col] = new_data_df[col].values

    # 4. Recalculer les features 'temps réel' qui dépendent des nouvelles données
    # Ce sont les ratios que tu avais créés dans 'application_train_test'
    precalculated_features['DAYS_EMPLOYED_ANOM'] = (precalculated_features['DAYS_EMPLOYED'] == 365243)
    precalculated_features['DAYS_EMPLOYED_PERC'] = precalculated_features['DAYS_EMPLOYED'] / precalculated_features[
        'DAYS_BIRTH']
    precalculated_features['INCOME_CREDIT_PERC'] = precalculated_features['AMT_INCOME_TOTAL'] / (precalculated_features[
        'AMT_CREDIT']+ epsilon )
    precalculated_features['INCOME_PER_PERSON'] = precalculated_features['AMT_INCOME_TOTAL'] / precalculated_features[
        'CNT_FAM_MEMBERS']
    precalculated_features['ANNUITY_INCOME_PERC'] = precalculated_features['AMT_ANNUITY'] / (precalculated_features[
        'AMT_INCOME_TOTAL']+ epsilon )
    precalculated_features['PAYMENT_RATE'] = precalculated_features['AMT_ANNUITY'] / (precalculated_features[
        'AMT_CREDIT']+ epsilon )

    # 5. Renvoyer la ligne finale prête pour le modèle
    return precalculated_features