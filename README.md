# API et Dashboard de Scoring de Crédit

Ce projet a pour but de développer et de déployer un modèle de scoring de crédit sous la forme d'une API et d'un dashboard interactif.
- L'**API**, construite avec FastAPI et conteneurisée avec Docker, prédit la probabilité de défaut de paiement d'un client.
- Le **Dashboard**, construit avec Streamlit, permet aux utilisateurs d'interroger l'API et de visualiser les raisons d'une décision grâce à l'interprétabilité du modèle (SHAP).

Ce projet a été réalisé dans le cadre de ma formation de Data Scientist.

**Lien vers l'API déployée :** `[Mets ici l'URL de ton API sur Render]`
**Lien vers le Dashboard (si déployé) :** `[Lien vers le dashboard]`

---

## 🚀 Fonctionnalités

* **API de Scoring :** Endpoint de prédiction de score et de décision (prêt accepté/refusé).
* **API d'Interprétabilité :** Endpoint qui fournit les explications SHAP pour une décision donnée.
* **Dashboard Interactif :** Interface simple pour tester l'API et visualiser les scores et les explications SHAP.
* **Déploiement Conteneurisé :** L'API est entièrement conteneurisée avec Docker pour un déploiement facile et reproductible sur le cloud.

---

## 📂 Structure du Projet
```
.
├── app/
│   └── ... (Code de l'API FastAPI)
├── dashboard/
│   └── app.py           # Code du Dashboard Streamlit
├── data/
│   └── feature_store.db # Base de données SQLite de production
├── model/
│   └── model.pkl        # Modèle LightGBM final
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
    git clone [https://github.com/TegroTON/TON-DEX-TegroFinance-Web-Frontend](https://github.com/TegroTON/TON-DEX-TegroFinance-Web-Frontend)
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

### Lancement du Dashboard

Le dashboard se connecte à l'API (qu'elle soit lancée en local ou sur le cloud).

1.  Assurez-vous que votre environnement virtuel est activé.
2.  Lancez l'application Streamlit :
    ```bash
    streamlit run dashboard/app.py
    ```
Le dashboard sera accessible à l'adresse `http://localhost:8501`.

---

## 📖 Endpoints de l'API

Une documentation interactive complète est disponible à l'adresse de l'API, sur le chemin `/docs` (par exemple, `http://localhost:8000/docs`).

### `POST /predict`
* **Description** : Prédit le risque de défaut pour un client.
* **Réponse en cas de succès** :
    ```json
    {
      "prediction": 0,
      "score": 0.0336
    }
    ```

### `GET /shap_explanation/{client_id}`
* **Description** : Fournit les données d'interprétabilité SHAP pour un client donné.
* **Réponse en cas de succès** : Un objet JSON contenant les valeurs de base, les valeurs SHAP et les noms des features.

---
*Projet réalisé par Thomas*