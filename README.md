# API de Scoring de Crédit avec Monitoring MLOps

Ce projet a pour but de développer et de déployer un modèle de scoring de crédit sous la forme d'une API conteneurisée, et de mettre en place un système de monitoring pour surveiller la dérive des données en production.

Ce projet a été réalisé dans le cadre de ma formation de Data Scientist.

**Lien vers l'API déployée :** `https://scoring-api-thomas.onrender.com`

---

## 🚀 Fonctionnalités

* **API de Scoring :** Endpoint de prédiction de score et de décision (prêt accepté/refusé).
* **API d'Interprétabilité :** Endpoint qui fournit les explications SHAP pour une décision donnée.
* **Monitoring de Modèle :** Un script utilisant Evidently AI permet de générer un rapport sur la dérive des données entre l'entraînement et la production.
* **Déploiement Conteneurisé :** L'API est entièrement conteneurisée avec Docker pour un déploiement facile et reproductible sur le cloud.

---

## 📂 Structure du Projet
```
.
├── app/
│   └── ... (Code de l'API FastAPI)
├── data/
│   └── feature_store.db # Base de données SQLite de production
├── model/
│   └── model.pkl        # Modèle LightGBM final
├── monitoring/
│   └── generate_report.py # Script de monitoring avec Evidently AI
├── notebooks/
│   └── ... (Notebooks d'analyse et de modélisation)
├── scripts/
│   └── convert_to_sqlite.py # Script de préparation des données
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── requirements-dev.txt
```


---

## ⚙️ Installation (Pour Développement)

Pour travailler sur le projet, suivez ces étapes :

1.  **Clonez le dépôt :**
    ```bash
    git clone [https://github.com/depot](https://github.com/depot)
    cd [nom-du-dossier]
    ```

2.  **Créez et activez un environnement virtuel :**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
    ```

3.  **Installez toutes les dépendances de développement :**
    ```bash
    pip install -r requirements-dev.txt
    ```

---

## ▶️ Utilisation

### Lancement de l'API avec Docker (Recommandé)

C'est la méthode de production. Assurez-vous d'avoir Docker Desktop d'installé et lancé.

1.  **Construisez l'image Docker :**
    ```bash
    docker build -t scoring-api .
    ```

2.  **Lancez le conteneur :**
    ```bash
    docker run -p 8000:8000 scoring-api
    ```
L'API sera accessible à l'adresse `http://localhost:8000`.

---
## 📈 Monitoring

Pour générer le rapport de dérive des données, assurez-vous d'avoir d'abord généré des données de production en utilisant l'API, puis d'avoir rapatrié le fichier `predictions_log.csv` dans le dossier `data/`.

1.  Assurez-vous que votre environnement virtuel est activé.
2.  Lancez le script de monitoring :
    ```bash
    python monitoring/generate_report.py
    ```
3.  Un rapport `data_drift_report.html` sera généré à la racine du projet.

---

## 📖 Endpoints de l'API

Une documentation interactive complète est disponible à l'adresse de l'API, sur le chemin `/docs` (par exemple, `http://localhost:8000/docs`).

### `POST /predict`
* **Description** : Prédit le risque de défaut pour un client.

### `GET /shap_explanation/{client_id}`
* **Description** : Fournit les données d'interprétabilité SHAP pour un client donné.

### `GET /download_logs`
* **Description** : Endpoint de maintenance pour télécharger le fichier de log des prédictions.

---
*Projet réalisé par [Votre Nom]*