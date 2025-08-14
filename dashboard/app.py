# dashboard/app.py (Version finale avec graphiques SHAP)
import streamlit as st
import requests
import pandas as pd
import json
import shap
import numpy as np
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Dashboard de Scoring Client",
    page_icon="🤖",
    layout="wide"  # On passe en layout large pour mieux voir les graphiques
)

# URLs de l'API
API_URL_PREDICT = "https://scoring-api-thomas.onrender.com/predict"
API_URL_SHAP = "https://scoring-api-thomas.onrender.com/shap_explanation"


# --- Fonctions Utilitaires ---
def get_shap_plot(client_id):
    """
    Appelle l'API pour récupérer les données SHAP et génère un graphique waterfall.
    """
    try:
        response = requests.get(f"{API_URL_SHAP}/{client_id}", timeout=30)
        response.raise_for_status()
        shap_data = response.json()

        # Recréer l'objet Explanation de SHAP
        shap_explanation = shap.Explanation(
            values=np.array(shap_data["shap_values"]),
            base_values=shap_data["base_value"],
            data=np.array(shap_data["feature_values"]),
            feature_names=shap_data["feature_names"]
        )

        st.subheader("🔎 Analyse détaillée de la Décision")
        st.markdown(
            """
            Le graphique ci-dessous (waterfall plot) montre comment chaque caractéristique
            du client a contribué au score final. On part du score moyen (en bas) et chaque 
            facteur (en rouge ou en bleu) fait évoluer le score jusqu'au résultat final (en haut).
            """
        )

        # --- C'est ici qu'on utilise la nouvelle méthode ---
        # On crée le graphique waterfall

        shap.plots.waterfall(shap_explanation, max_display=15, show=False)

        # On récupère la figure matplotlib actuelle et on l'affiche dans Streamlit
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)  # Important pour libérer la mémoire

    except requests.exceptions.RequestException as e:
        st.warning(f"Impossible de générer l'explication SHAP : {e}")


# --- Interface du Dashboard ---
st.title("👨‍💻 Dashboard de Scoring Client")
st.markdown(
    """
    Ce dashboard permet d'interroger notre API de scoring pour prédire la probabilité 
    de défaut de paiement d'un client et de comprendre la décision du modèle.
    """
)
st.divider()

st.header("Informations du client et de la demande de prêt")
col1, col2 = st.columns(2)
with col1:
    client_id = st.text_input(label="**Identifiant du client**", placeholder="Ex: 100002")
    # ... (les autres number_input ne changent pas)
    amt_income = st.number_input(label="**Revenu total (€)**", min_value=0, value=202500, step=1000)
    amt_credit = st.number_input(label="**Montant du crédit (€)**", min_value=0, value=406597, step=1000)
with col2:
    amt_annuity = st.number_input(label="**Montant de l'annuité (€)**", min_value=0, value=24700, step=500)
    days_birth = st.number_input(label="**Âge (en jours)**", max_value=0, value=-9461, step=1)
    days_employed = st.number_input(label="**Ancienneté emploi (en jours)**", max_value=0, value=-637, step=1)

if st.button("Obtenir la prédiction", type="primary"):
    if client_id:
        payload = {"SK_ID_CURR": int(client_id), "AMT_CREDIT": amt_credit, "AMT_INCOME_TOTAL": amt_income,
                   "AMT_ANNUITY": amt_annuity, "DAYS_BIRTH": days_birth, "DAYS_EMPLOYED": days_employed}
        with st.spinner('Appel à l\'API en cours...'):
            try:
                response = requests.post(API_URL_PREDICT, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                score = result.get("score")
                prediction = result.get("prediction")

                st.divider()
                st.header("Résultat de la prédiction")
                if prediction == 0:
                    st.success("✅ **Prêt Accepté**")
                else:
                    st.error("❌ **Prêt Refusé**")

                st.metric(label="**Score de risque**", value=f"{score:.2%}")
                st.progress(score, text=f"Probabilité de défaut : {score:.2%}")

                # --- Appel à la fonction pour afficher le graphique SHAP ---
                get_shap_plot(int(client_id))

            except requests.exceptions.HTTPError as err:
                st.error(f"Erreur HTTP : {err.response.status_code}")
                st.json(err.response.json())
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API : {e}")
    else:
        st.error("⚠️ Veuillez saisir un identifiant de client.")