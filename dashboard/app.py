# dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Scoring Client",
    page_icon="🤖",
    layout="wide"
)

# --- URLs de l'API et des données ---
API_URL_PREDICT = "https://scoring-api-thomas.onrender.com/predict"


# --- Fonctions Utilitaires ---
@st.cache_data
def load_data():
    """Charge l'échantillon de données clients depuis GitHub."""
    url = 'https://github.com/tmoahs/Scoring-API-P7/releases/download/v3-data/application_train_sample.parquet'
    data = pd.read_parquet(url)
    return data


@st.cache_resource  # Utiliser cache_resource pour les objets complexes
def load_shap_data():
    """Charge les objets SHAP pré-calculés depuis GitHub."""
    url_explanation = "https://github.com/tmoahs/Scoring-API-P7/releases/download/v3-data/shap_explanation_object.pkl"
    url_sample = "https://github.com/tmoahs/Scoring-API-P7/releases/download/v3-data/shap_data_sample.pkl"

    try:
        explanation_object = pd.read_pickle(url_explanation)
        X_sample = pd.read_pickle(url_sample)
        return explanation_object, X_sample
    except Exception as e:
        st.error(f"Erreur lors du chargement des données SHAP depuis GitHub. Vérifiez les URL. Erreur : {e}")
        return None, None


def create_gauge_chart(score):
    score_percent = score * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score de Risque du Client", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgrey"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(40, 167, 69, 0.7)'},
                {'range': [40, 50], 'color': 'rgba(255, 193, 7, 0.7)'},
                {'range': [50, 100], 'color': 'rgba(220, 53, 69, 0.7)'},

                {'range': [max(0, score_percent - 1), min(100, score_percent + 1)], 'color': 'black'}
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"},
        height=300
    )
    return fig


# --- Chargement des données au démarrage ---
data = load_data()
explanation_object, X_sample = load_shap_data()

# --- Initialisation du Session State (la "mémoire" de l'app) ---
if 'prediction_faite' not in st.session_state:
    st.session_state.prediction_faite = False
    st.session_state.client_id = None
    st.session_state.score = None
    st.session_state.prediction = None

# --- Création des colonnes pour la mise en page ---
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # --- Interface du Dashboard ---
    st.title("Scoring Client")
    st.markdown(
        "Ce dashboard permet de prédire la probabilité de défaut de paiement d'un client et de comprendre la décision.")
    st.divider()
    # --- Section de saisie des informations ---
    st.header("Informations du client")

    valid_ids = data['SK_ID_CURR'].head(10).tolist()
    st.info(f"Astuce : Essayez avec un de ces ID clients présents dans l'échantillon : {valid_ids}")

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        client_id_input = st.text_input(label="**Identifiant du client**", value=str(valid_ids[0]))
        amt_income = st.number_input(label="**Revenu total (€)**", min_value=0, value=202500, step=1000)
        amt_credit = st.number_input(label="**Montant du crédit (€)**", min_value=0, value=406597, step=1000)
    with input_col2:
        amt_annuity = st.number_input(label="**Montant de l'annuité (€)**", min_value=0, value=24700, step=500)
        days_birth = st.number_input(label="**Âge (en jours)**", max_value=0, value=-9461, step=1)
        days_employed = st.number_input(label="**Ancienneté emploi (en jours)**", max_value=0, value=-637, step=1)

    # --- Bouton de prédiction ---
    if st.button("Analyser le client", type="primary"):
        if client_id_input:
            payload = {"SK_ID_CURR": int(client_id_input), "AMT_CREDIT": amt_credit, "AMT_INCOME_TOTAL": amt_income,
                       "AMT_ANNUITY": amt_annuity, "DAYS_BIRTH": days_birth, "DAYS_EMPLOYED": days_employed}
            with st.spinner('Analyse en cours...'):
                try:
                    response = requests.post(API_URL_PREDICT, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()

                    st.session_state.prediction_faite = True
                    st.session_state.client_id = int(client_id_input)
                    st.session_state.score = result.get("score")
                    st.session_state.prediction = result.get("prediction")

                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur de connexion à l'API : {e}")
                    st.session_state.prediction_faite = False
        else:
            st.error("⚠️ Veuillez saisir un identifiant de client.")

    # --- Affichage des résultats (conditionné par le session_state) ---
    if st.session_state.prediction_faite:
        st.divider()
        st.header("Résultat de la prédiction")

        col1_res, col2_res = st.columns([1, 2])  # Crée deux colonnes

        with col1_res:
            if st.session_state.prediction == 0:
                st.success("**Prêt Accepté**")
            else:
                st.error("**Prêt Refusé**")

            # On affiche le score et la probabilité ici, en texte
            st.metric(label="**Score de risque**", value=f"{st.session_state.score:.2%}")
            st.write("Ce score représente la probabilité de défaut du client.")

        with col2_res:
            # On affiche la jauge sans le chiffre dans la deuxième colonne
            st.plotly_chart(create_gauge_chart(st.session_state.score), use_container_width=True)

        # --- Section d'Analyse SHAP ---
        st.divider()
        st.header("Analyse détaillée de la Décision")

        # Vérifie si les données SHAP ont bien été chargées
        if explanation_object is not None and X_sample is not None:
            col_shap_1, col_shap_2 = st.columns(2)
            with col_shap_1:
                st.subheader("Importance des caractéristiques locales")
                st.markdown("Impact pour le client sélectionné.")
                try:
                    client_index = X_sample.index.get_loc(st.session_state.client_id)
                    fig_shap_local, ax_shap_local = plt.subplots()
                    shap.plots.waterfall(explanation_object[client_index], max_display=15, show=False)
                    st.pyplot(fig_shap_local)
                    plt.close(fig_shap_local)
                except KeyError:
                    st.warning("Client non trouvé dans l'échantillon SHAP pour l'analyse locale.")

            with col_shap_2:
                st.subheader("Importance des caractéristiques globales")
                st.markdown("Impact moyen sur l'ensemble des clients.")
                fig_shap_global, ax_shap_global = plt.subplots()
                shap.summary_plot(explanation_object.values, X_sample, plot_type="bar", max_display=15, show=False)
                plt.tight_layout()
                st.pyplot(fig_shap_global)
                plt.close(fig_shap_global)
        else:
            st.warning("Les données d'analyse SHAP n'ont pas pu être chargées.")

        # --- Section d'Analyse Comparative (Version corrigée et finale) ---
        st.divider()
        st.header("Analyse comparative du client")

        if st.session_state.client_id in data['SK_ID_CURR'].values:

            # Dictionnaire pour avoir des noms plus clairs dans les menus déroulants
            feature_labels = {
                'AMT_INCOME_TOTAL': 'Revenu total',
                'AMT_CREDIT': 'Montant du crédit',
                'AMT_ANNUITY': 'Montant de l\'annuité',
                'AMT_GOODS_PRICE': 'Prix du bien',
                'DAYS_BIRTH': 'Âge',
                'DAYS_EMPLOYED': 'Ancienneté emploi',
                'CNT_CHILDREN': 'Nombre d\'enfants',
                'EXT_SOURCE_1': 'Score externe 1',
                'EXT_SOURCE_2': 'Score externe 2',
                'EXT_SOURCE_3': 'Score externe 3'
            }

            # On garde les clés (noms techniques) pour le code
            feature_list = list(feature_labels.keys())

            # On peut aussi filtrer pour le scatter plot si besoin
            continuous_feature_list = [
                'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
            ]

            # --- Graphique de distribution (univarié) ---
            st.subheader("Distribution d'une caractéristique")
            selected_feature_dist = st.selectbox(
                'Sélectionnez une caractéristique :',
                options=feature_list,
                # On utilise le dictionnaire pour afficher le nom clair
                format_func=lambda x: feature_labels[x],
                index=feature_list.index('AMT_INCOME_TOTAL')
            )

            # Logique de filtrage et d'affichage pour le graphique de distribution
            client_value = data.loc[data['SK_ID_CURR'] == st.session_state.client_id, selected_feature_dist].values[0]
            percentile_99 = data[selected_feature_dist].quantile(0.99)
            plot_data_dist = data[data[selected_feature_dist] < percentile_99]

            fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
            sns.histplot(plot_data_dist[selected_feature_dist].dropna(), ax=ax_dist, kde=True, color="skyblue",
                         label="Tous les clients (hors 1% extrêmes)")

            if client_value < percentile_99:
                ax_dist.axvline(x=client_value, color='red', linestyle='--', linewidth=2,
                                label=f'Client {st.session_state.client_id}')
            else:
                st.warning(
                    f"Le client est une valeur extrême pour cette caractéristique et n'est pas affiché sur le graphique zoomé.")

            titre_dist = feature_labels.get(selected_feature_dist, selected_feature_dist)
            ax_dist.set_title(f'Distribution de "{titre_dist}"')
            ax_dist.legend()
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close(fig_dist)

            # --- Graphique bi-varié ---
            st.subheader("Analyse bi-variée")
            col_bi_1, col_bi_2 = st.columns(2)
            with col_bi_1:
                feature_x = st.selectbox("Caractéristique (Axe X) :",
                                         options=continuous_feature_list,
                                         format_func=lambda x: feature_labels.get(x, x),
                                         index=continuous_feature_list.index('AMT_CREDIT'))
            with col_bi_2:
                feature_y = st.selectbox("Caractéristique (Axe Y) :",
                                         options=continuous_feature_list,
                                         format_func=lambda x: feature_labels.get(x, x),
                                         index=continuous_feature_list.index('AMT_ANNUITY'))

            # Logique de filtrage et d'affichage pour le graphique bi-varié
            data_display = data.copy()
            data_display['TARGET'] = data_display['TARGET'].replace({0: 'Prêt Accepté', 1: 'Prêt Refusé'})

            percentile_99_x = data_display[feature_x].quantile(0.99)
            percentile_99_y = data_display[feature_y].quantile(0.99)

            plot_data_bi = data_display[
                (data_display[feature_x] < percentile_99_x) &
                (data_display[feature_y] < percentile_99_y)
                ]

            client_x_val = data.loc[data['SK_ID_CURR'] == st.session_state.client_id, feature_x].values[0]
            client_y_val = data.loc[data['SK_ID_CURR'] == st.session_state.client_id, feature_y].values[0]

            fig_bi, ax_bi = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=plot_data_bi, x=feature_x, y=feature_y, hue='TARGET', style='TARGET', alpha=0.5,
                            ax=ax_bi, palette='colorblind')

            if client_x_val < percentile_99_x and client_y_val < percentile_99_y:
                ax_bi.scatter(client_x_val, client_y_val, color='red', s=100, edgecolor='black',
                              label=f'Client {st.session_state.client_id}', zorder=3)
            else:
                st.warning("Le client sélectionné est une valeur extrême et n'est pas affiché sur ce graphique zoomé.")

            titre_x = feature_labels.get(feature_x, feature_x)
            titre_y = feature_labels.get(feature_y, feature_y)
            ax_bi.set_title(f'Relation entre "{titre_x}" et "{titre_y}"')
            ax_bi.legend(title='Statut du Prêt')
            plt.tight_layout()
            st.pyplot(fig_bi)
            plt.close(fig_bi)

        else:
            st.warning("Client non trouvé dans l'échantillon pour la comparaison.")