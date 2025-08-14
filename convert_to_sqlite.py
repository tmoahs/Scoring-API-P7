# convert_to_sqlite.py
import pandas as pd
import sqlite3
import os

print("Début de la conversion Parquet vers SQLite...")

# Définir les chemins
parquet_path = 'data/dataset_optimized.parquet' # Le nom de ton fichier optimisé
db_path = 'data/feature_store.db'
table_name = 'features'

# 1. Charger le DataFrame depuis le fichier Parquet
print(f"Chargement de {parquet_path}...")
df = pd.read_parquet(parquet_path)
print("DataFrame chargé.")

# 2. Créer une connexion à la base de données SQLite
print(f"Création de la base de données à l'adresse {db_path}...")
conn = sqlite3.connect(db_path)

# 3. Écrire le DataFrame dans une table SQL
print(f"Écriture des données dans la table '{table_name}'...")
df.to_sql(table_name, conn, if_exists='replace', index=False)
print("Données écrites avec succès.")

# 4. CRUCIAL : Créer un index sur la colonne SK_ID_CURR
# C'est ce qui rendra les recherches ultra-rapides.
print(f"Création d'un index sur la colonne SK_ID_CURR...")
conn.execute(f'CREATE INDEX idx_sk_id_curr ON {table_name} (SK_ID_CURR);')
print("Index créé.")

# 5. Fermer la connexion
conn.close()

print(f"✅ Conversion terminée. Le fichier '{db_path}' est prêt !")