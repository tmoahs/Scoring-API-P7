# app/preprocessing.py (version avec lecture à la demande)
import pandas as pd
import numpy as np
import os


def prepare_data_for_prediction(client_id: int, new_loan_data: dict, data_path: str) -> pd.DataFrame:
    """
    Prépare la ligne de données finale pour un client donné en la lisant
    directement depuis le fichier Parquet au moment de la requête.
    """
    try:
        # 1. Lire UNIQUEMENT la ligne du client demandé depuis le fichier Parquet
        # La fonction 'filters' de read_parquet est très efficace pour cela.
        client_features = pd.read_parquet(
            data_path,
            filters=[[('SK_ID_CURR', '==', client_id)]]
        )
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la lecture du fichier Parquet : {e}")

    if client_features.empty:
        raise ValueError(f"Client avec SK_ID_CURR {client_id} non trouvé dans le fichier de données.")

    # 2. Mettre à jour avec les données de la nouvelle requête
    new_data_df = pd.DataFrame([new_loan_data])
    for col in new_data_df.columns:
        if col in client_features.columns:
            # .loc est plus sûr pour éviter les avertissements de copie
            client_features.loc[:, col] = new_data_df[col].values

    # 3. Recalculer les features 'temps réel' (si nécessaire)
    # Tu peux ajouter ici les calculs de ratio comme INCOME_CREDIT_PERC si besoin

    return client_features