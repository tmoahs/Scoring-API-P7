# dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import json

# Configuration de la page
st.set_page_config(
    page_title="Dashboard de Scoring Client",
    page_icon="🤖",
    layout="centered"
)

# URL de l'API déployée sur Render
API_URL = "https://scoring-api-thomas.onrender.com/predict"

# Titre du dashboard
st.title("👨‍💻 Dashboard de Scoring Client")

# Introduction
st.markdown(
    """
    Ce dashboard permet d'interroger notre API de scoring pour prédire la probabilité 
    de défaut de paiement d'un client. Saisissez les informations ci-dessous pour obtenir une prédiction.
    """
)

st.divider()

# --- Formulaire de saisie ---
st.header("Informations du client et de la demande de prêt")

# Créer des colonnes pour une meilleure disposition
col1, col2 = st.columns(2)

with col1:
    client_id = st.text_input(
        label="**Identifiant du client**",
        placeholder="Ex: 100002"
    )
    amt_income = st.number_input(
        label="**Revenu total (€)**",
        min_value=0,
        value=202500,
        step=1000,
        help="Revenu annuel total du client."
    )
    amt_credit = st.number_input(
        label="**Montant du crédit demandé (€)**",
        min_value=0,
        value=406597,
        step=1000,
        help="Montant total du prêt demandé."
    )

with col2:
    amt_annuity = st.number_input(
        label="**Montant de l'annuité (€)**",
        min_value=0,
        value=24700,
        step=500,
        help="Montant de l'annuité du prêt."
    )
    days_birth = st.number_input(
        label="**Âge (en jours)**",
        max_value=0,
        value=-9461,
        step=1,
        help="Âge du client en jours au moment de la demande. Doit être négatif."
    )
    days_employed = st.number_input(
        label="**Ancienneté emploi (en jours)**",
        max_value=0,
        value=-637,
        step=1,
        help="Nombre de jours d'emploi avant la demande. Doit être négatif."
    )

# --- Bouton et Logique de Prédiction ---
if st.button("Obtenir la prédiction", type="primary"):
    if client_id:
        # Création du payload (les données à envoyer à l'API)
        payload = {
            "SK_ID_CURR": int(client_id),
            "AMT_CREDIT": amt_credit,
            "AMT_INCOME_TOTAL": amt_income,
            "AMT_ANNUITY": amt_annuity,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed
            # NOTE: Tu peux ajouter d'autres champs ici si ton API les utilise
        }

        # Affichage d'un message d'attente
        with st.spinner('Appel à l\'API en cours...'):
            try:
                # Appel à l'API
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()  # Lève une exception si le statut est une erreur (4xx ou 5xx)

                # Traitement de la réponse
                result = response.json()
                score = result.get("score")
                prediction = result.get("prediction")

                st.divider()
                st.header("Résultat de la prédiction")

                if prediction == 0:
                    st.success("✅ **Prêt Accepté**")
                else:
                    st.error("❌ **Prêt Refusé**")

                # Affichage du score avec une jauge
                st.metric(label="**Score de risque**", value=f"{score:.2%}")
                st.progress(score, text=f"Probabilité de défaut : {score:.2%}")
                st.caption("Plus le score est bas, moins le client est risqué.")

            except requests.exceptions.HTTPError as err:
                st.error(f"Erreur HTTP : {err.response.status_code}")
                st.json(err.response.json())  # Affiche le détail de l'erreur renvoyée par l'API
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API : {e}")

    else:
        st.error("⚠️ Veuillez saisir un identifiant de client.")