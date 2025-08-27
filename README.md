# Projet de Scoring de Crédit : API, Monitoring & Dashboard Interactif

Ce projet complet de Data Science vise à développer un modèle de scoring de crédit, à le déployer via une **API RESTful conteneurisée**, à surveiller sa performance en production via un **rapport de monitoring**, et à le rendre accessible aux utilisateurs métier grâce à un **dashboard interactif**.

L'objectif final est de prédire la probabilité de défaut de paiement d'un client et de fournir des outils pour interpréter cette décision de manière transparente.

Ce projet a été réalisé dans le cadre de ma formation de Data Scientist.

---

## 🚀 Composants du Projet & Liens

Ce projet est divisé en deux applications distinctes et un processus de surveillance :

1.  **Le Dashboard Interactif (Frontend)** : Une application Streamlit pour les chargés de clientèle.
    * **➡️ Lien vers l'application déployée :** `https://scoring-api-p8.streamlit.app/`

2.  **L'API de Scoring (Backend)** : Une API FastAPI qui expose le modèle de prédiction.
    * **➡️ Lien vers l'API déployée :** `https://scoring-api-thomas.onrender.com`

3.  **Le Monitoring MLOps** : Un rapport généré avec Evidently AI pour surveiller la dérive des données.

---

## 📊 1. Dashboard Interactif de Scoring

Le dashboard est l'interface utilisateur principale, conçue pour permettre aux chargés de clientèle d'expliquer les décisions de crédit de manière simple et visuelle.

### Fonctionnalités du Dashboard

* **Score Client** : Visualisation du score de risque via une jauge intuitive et affichage de la décision (prêt accepté/refusé).
* **Interprétabilité du Modèle** : Affichage des facteurs les plus influents pour la décision d'un client (SHAP local) et comparaison avec l'importance globale des facteurs.
* **Analyse Comparative** : Graphiques interactifs (distributions et nuages de points) permettant de comparer un client à l'ensemble de la clientèle.
* **Simulation "What-If"** : Possibilité de modifier les informations d'un client existant pour voir l'impact en temps réel sur son score.
* **Accessibilité** : Conception des graphiques pensée pour l'accessibilité (palettes de couleurs, marqueurs multiples).

### Lancement Local du Dashboard

1.  Assurez-vous que l'API de scoring (voir ci-dessous) est lancée et accessible.
2.  Naviguez vers le dossier du dashboard (`cd dashboard/`).
3.  Installez les dépendances : `pip install -r requirements.txt`.
4.  Lancez l'application Streamlit :
    ```bash
    streamlit run app.py
    ```

---

## ⚙️ 2. API de Scoring de Crédit

L'API est le cœur technique du projet, responsable du calcul des prédictions et de la fourniture des données d'interprétabilité.

### Fonctionnalités de l'API

* **Prédiction de score** : Prédit la probabilité de défaut pour un client donné.
* **Explication SHAP** : Fournit les données nécessaires pour générer les graphiques d'interprétabilité.
* **Déploiement Conteneurisé** : Entièrement conteneurisée avec Docker pour un déploiement facile.
* **Documentation automatique** : Documentation interactive disponible via Swagger UI au endpoint `/docs`.

### Lancement de l'API avec Docker (Recommandé)

Assurez-vous d'avoir Docker Desktop d'installé et lancé.

1.  **Construisez l'image Docker** (depuis la racine du projet) :
    ```bash
    docker build -t scoring-api .
    ```
2.  **Lancez le conteneur :**
    ```bash
    docker run -p 8000:8000 scoring-api
    ```

---

## 📈 3. Monitoring

Pour générer le rapport de dérive des données, lancez le script suivant depuis la racine du projet (environnement virtuel activé) :
```bash
python monitoring/generate_report.py
```

---

## 📂 Structure du Projet
```
.
├── app/ (Code de l'API FastAPI)
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── preprocessing.py
├── model/
│   └── model.pkl # Modèle final    
├── dashboard/            # Code source du Dashboard Streamlit
│   ├── app.py
│   └── requirements.txt
├── monitoring/
│   └── generate_report.py # Script de monitoring avec Evidently AI
├── notebooks/
│   ├── EDA + FE.ipynb
│   ├── MODELISATION.ipynb
│   └── Analyse Modèle.ipynb
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
*Projet réalisé par Thomas*