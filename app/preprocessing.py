# app/preprocessing.py (Version finale avec base de données SQLite)
import pandas as pd
import sqlite3


def prepare_data_for_prediction(client_id: int, new_loan_data: dict, db_path: str) -> pd.DataFrame:
    """
    Prépare la ligne de données finale pour un client donné en l'interrogeant
    directement depuis la base de données SQLite.
    """
    conn = None  # Initialiser la connexion à None
    try:
        # 1. Se connecter à la base de données SQLite
        conn = sqlite3.connect(db_path)

        # 2. Écrire la requête SQL pour récupérer la ligne du client
        # L'utilisation de '?' est une protection contre les injections SQL.
        query = f"SELECT * FROM features WHERE SK_ID_CURR = ?"

        # 3. Exécuter la requête avec pandas
        client_features = pd.read_sql_query(query, conn, params=(client_id,))

    except sqlite3.Error as e:
        raise RuntimeError(f"Erreur de base de données : {e}")
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue : {e}")
    finally:
        # 5. S'assurer que la connexion est toujours fermée, même en cas d'erreur
        if conn:
            conn.close()

    if client_features.empty:
        raise ValueError(f"Client avec SK_ID_CURR {client_id} non trouvé dans la base de données.")

    new_data_df = pd.DataFrame([new_loan_data])
    for col in new_data_df.columns:
        if col in client_features.columns:
            client_features.loc[:, col] = new_data_df[col].values

    return client_features