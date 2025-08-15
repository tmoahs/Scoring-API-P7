# monitoring/generate_report.py (Version finale qui gère les colonnes vides)
import pandas as pd
import os

from evidently import Report
from evidently.presets import DataDriftPreset

print("Début de la génération du rapport de monitoring...")

# --- Chemins d'accès robustes ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REF_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset_optimized.parquet')
PROD_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'predictions_log.csv')
# --- Fin ---


# 1. Charger les données de référence
try:
    reference_data = pd.read_parquet(REF_DATA_PATH).sample(n=5000, random_state=42)
    print(f"Données de référence chargées depuis '{REF_DATA_PATH}'.")
except FileNotFoundError:
    print(f"ERREUR: Le fichier '{REF_DATA_PATH}' est introuvable.")
    exit()

# 2. Charger les données de production
try:
    production_data = pd.read_csv(PROD_DATA_PATH)
    print(f"Données de production chargées depuis '{PROD_DATA_PATH}'.")
except FileNotFoundError:
    print(f"ERREUR: Le fichier '{PROD_DATA_PATH}' est introuvable.")
    exit()

# --- 3. PRÉPARATION FINALE DES DONNÉES ---
# On gère la colonne 'TARGET'
if 'TARGET' in reference_data.columns:
    reference_data = reference_data.drop(columns=['TARGET'])

# On identifie les colonnes entièrement vides dans les données de production
empty_cols = [col for col in production_data.columns if production_data[col].isnull().all()]
if empty_cols:
    print(f"AVERTISSEMENT: Les colonnes suivantes sont entièrement vides et seront exclues de l'analyse : {empty_cols}")
    # On les retire des deux jeux de données pour garder la cohérence
    reference_data = reference_data.drop(columns=empty_cols, errors='ignore')
    production_data = production_data.drop(columns=empty_cols)

# On aligne les colonnes restantes
production_data_aligned = production_data[reference_data.columns]
print("Alignement des colonnes terminé.")

# --- Fin de la préparation ---

# 4. Créer le rapport Evidently
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

# 5. Lancer l'analyse
my_eval = data_drift_report.run(reference_data=reference_data,current_data=production_data_aligned)

# 6. Sauvegarder le rapport
report_path = os.path.join(PROJECT_ROOT, 'data_drift_report.html')
my_eval.save_html(report_path)

print(f"✅ Rapport de dérive de données généré avec succès !")
print(f"Ouvrez le fichier '{report_path}' dans votre navigateur pour l'analyser.")