# create_prod_dataset.py (version optimisée pour la mémoire)
import pandas as pd
import numpy as np

print("Début du script d'optimisation de mémoire...")

# 1. Charger le jeu de données complet
print("Chargement du dataset complet...")
df = pd.read_parquet('data/final_dataset.parquet')
mem_usage_before = df.memory_usage(deep=True).sum() / 1024**2
print(f"Usage mémoire avant optimisation: {mem_usage_before:.2f} MB")

# 2. Optimiser les types de données
print("Optimisation des types de données (downcasting)...")
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype(np.float32)
    elif df[col].dtype == 'int64':
        # On vérifie si on peut réduire la taille de l'entier
        if df[col].min() >= np.iinfo(np.int8).min and df[col].max() <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)

mem_usage_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"Usage mémoire après optimisation: {mem_usage_after:.2f} MB")
print(f"Réduction de la mémoire de {((mem_usage_before - mem_usage_after) / mem_usage_before) * 100:.2f}%")

# 3. Sauvegarder le nouveau fichier Parquet optimisé
nouveau_nom_fichier = 'data/dataset_optimized.parquet'
print(f"Sauvegarde du dataset optimisé dans '{nouveau_nom_fichier}'...")
df.to_parquet(nouveau_nom_fichier, index=False)

print("✅ Script terminé. Le fichier de production optimisé est prêt !")