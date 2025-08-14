# API de Scoring de Crédit

Ce projet a pour but de développer et de déployer un modèle de scoring de crédit sous la forme d'une API. L'API utilise un modèle LightGBM pour prédire la probabilité de défaut de paiement d'un client en fonction de son historique et des informations de sa demande de prêt.

Ce projet a été réalisé dans le cadre de ma formation de Data Scientist.

---

## 🚀 Fonctionnalités

* **Prédiction de score** : Prédit la probabilité de défaut pour un client donné.
* **Décision binaire** : Fournit une décision simple (prêt accepté/refusé) basée sur un seuil.
* **Dockerisée** : L'application est entièrement conteneurisée avec Docker pour un déploiement facile et reproductible.
* **Documentation automatique** : Une documentation interactive de l'API est disponible via l'interface Swagger UI.

---

## 📂 Structure du Projet
```
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── preprocessing.py
├── data/
│   └── final_dataset.parquet
├── model/
│   └── model.pkl
├── notebooks/
│   ├── EDA + FE.ipynb
│   ├── MODELISATION.ipynb
│   └── Analyse Modèle.ipynb
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
    git clone [https://github.com/tmoahs/Projet-7](https://github.com/tmoahs/Projet-7)
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

## ▶️ Utilisation de l'API

Il y a deux manières de lancer l'API.

### 1. Lancement Local (sans Docker)

Cette méthode est utile pour le développement et le débogage rapide.

```bash
uvicorn app.main:app --reload
```

### 2. Lancement avec Docker (Recommandé)
C'est la méthode de production. Assurez-vous d'avoir Docker Desktop d'installé et lancé.

**Construisez l'image Docker :**

```bash 
docker build -t scoring-api .
```

**Lancez le conteneur :**

```bash
docker run -p 8000:8000 scoring-api
```

L'API sera accessible à l'adresse http://localhost:8000.

## 📖 Endpoints de l'API

Une fois l'API lancée, une documentation interactive complète est disponible à l'adresse http://localhost:8000/docs.

`GET /`
**Description :** Endpoint racine pour vérifier que l'API est en ligne.

**Réponse :**

```bash
{
  "message": "API de scoring en ligne et fonctionnelle."
}
```

`POST /predict`
**Description :** Prédit le risque de défaut pour un client.

**Corps de la requête :** Un objet JSON avec les informations du client (voir l'exemple dans la documentation /docs).

**Réponse en cas de succès :**
```bash
{
  "prediction": 0,
  "score": 0.0336
}
```
    
`prediction` : `0` pour un prêt accepté, `1` pour un prêt refusé.

`score` : La probabilité de défaut estimée par le modèle.