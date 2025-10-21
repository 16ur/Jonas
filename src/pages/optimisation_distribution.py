"""
Page d'optimisation de la distribution des vaccins en pharmacie
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Jonas - Optimisation Distribution",
    page_icon="📊",
    layout="wide"
)

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/master_dataframe.csv')
    df['date_semaine'] = pd.to_datetime(df['date_semaine'])
    return df

data = load_data()

# Header
st.title("Optimisation de la Distribution des Vaccins")
st.markdown("**Allocation intelligente des doses par région basée sur les besoins réels**")
st.markdown("---")

# Sidebar - Paramètres
st.sidebar.header("Paramètres d'optimisation")

doses_disponibles = st.sidebar.number_input(
    "Doses disponibles",
    min_value=0,
    max_value=10000000,
    value=1000000,
    step=100000,
    help="Nombre total de doses de vaccins à distribuer"
)

objectif_couverture = st.sidebar.slider(
    "Objectif de couverture (%)",
    min_value=50,
    max_value=95,
    value=75,
    step=5,
    help="Objectif de couverture vaccinale (OMS recommande 75%)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Stratégie de distribution")

strategie = st.sidebar.radio(
    "Choisissez votre priorité",
    options=["Équilibré", "Prévention", "Gestion de crise", "Anticipation"],
    index=0,
    help="Sélectionnez la stratégie qui correspond le mieux à vos objectifs"
)

# Description de la stratégie choisie
if strategie == "Équilibré":
    st.sidebar.info("""
    **Approche équilibrée**

    Prend en compte tous les critères de manière égale :
    - Régions avec faible couverture vaccinale
    - Régions avec forte activité aux urgences
    - Régions avec tendance à la hausse
    """)
    poids_deficit = 0.33
    poids_urgences = 0.33
    poids_tendance = 0.34

elif strategie == "Prévention":
    st.sidebar.success("""
    **Focus sur la prévention**

    Priorité aux régions ayant la plus faible couverture vaccinale pour éviter les futures épidémies.

    (Idéal pour campagnes de vaccination)
    """)
    poids_deficit = 0.70
    poids_urgences = 0.15
    poids_tendance = 0.15

elif strategie == "Gestion de crise":
    st.sidebar.error("""
    **Réponse d'urgence**

    Priorité aux régions actuellement en crise avec forte activité aux urgences.

    (Idéal en pleine épidémie)
    """)
    poids_deficit = 0.15
    poids_urgences = 0.70
    poids_tendance = 0.15

else:  # Anticipation
    st.sidebar.warning("""
    **Anticipation des pics**

    Priorité aux régions montrant des signes d'augmentation pour prévenir les pics.

    (Idéal en début de saison grippale) 
    """)
    poids_deficit = 0.15
    poids_urgences = 0.15
    poids_tendance = 0.70

# Données de la dernière semaine
latest_date = data['date_semaine'].max()
latest_data = data[data['date_semaine'] == latest_date].copy()

# Calculer la tendance (4 dernières semaines)
last_4_weeks = data[data['date_semaine'] >= (latest_date - pd.Timedelta(days=28))]
tendance_by_region = last_4_weeks.groupby('region').apply(
    lambda x: x.sort_values('date_semaine')['urgences_grippe'].diff().mean()
).reset_index(name='tendance_urgences')

# Fusionner avec les données actuelles
latest_data = latest_data.merge(tendance_by_region, on='region', how='left')

# Calculer les métriques pour l'optimisation
latest_data['deficit_couverture'] = objectif_couverture - latest_data['vacc_65_plus'].fillna(0)
latest_data['deficit_couverture'] = latest_data['deficit_couverture'].clip(lower=0)

# Normaliser les métriques (0-1)
latest_data['deficit_norm'] = (latest_data['deficit_couverture'] - latest_data['deficit_couverture'].min()) / (
    latest_data['deficit_couverture'].max() - latest_data['deficit_couverture'].min() + 0.001
)

latest_data['urgences_norm'] = (latest_data['urgences_grippe'] - latest_data['urgences_grippe'].min()) / (
    latest_data['urgences_grippe'].max() - latest_data['urgences_grippe'].min() + 0.001
)

latest_data['tendance_norm'] = (latest_data['tendance_urgences'] - latest_data['tendance_urgences'].min()) / (
    latest_data['tendance_urgences'].max() - latest_data['tendance_urgences'].min() + 0.001
)

# Remplacer les NaN par 0
latest_data['tendance_norm'] = latest_data['tendance_norm'].fillna(0)

# Calcul du score de priorité
latest_data['score_priorite'] = (
    poids_deficit * latest_data['deficit_norm'] +
    poids_urgences * latest_data['urgences_norm'] +
    poids_tendance * latest_data['tendance_norm']
)

# Calcul de l'allocation (proportionnelle au score)
total_score = latest_data['score_priorite'].sum()
if total_score > 0:
    latest_data['doses_allouees'] = (latest_data['score_priorite'] / total_score * doses_disponibles).round(0).astype(int)
else:
    latest_data['doses_allouees'] = 0

# Catégorie de priorité
latest_data['categorie'] = pd.cut(
    latest_data['score_priorite'],
    bins=[0, 0.3, 0.6, 0.8, 1.0],
    labels=['Faible', 'Moyenne', 'Élevée', 'Critique']
)

# Trier par score de priorité
latest_data = latest_data.sort_values('score_priorite', ascending=False)

# Affichage des résultats
st.markdown("### Résultats de l'optimisation")

col_result1, col_result2, col_result3, col_result4 = st.columns(4)

with col_result1:
    st.metric(
        "Doses allouées",
        f"{latest_data['doses_allouees'].sum():,.0f}",
        delta=f"{(latest_data['doses_allouees'].sum() / doses_disponibles * 100):.1f}%"
    )

with col_result2:
    nb_regions_prioritaires = len(latest_data[latest_data['categorie'].isin(['Critique', 'Élevée'])])
    st.metric(
        "Régions prioritaires",
        nb_regions_prioritaires,
        delta="Critique + Élevée"
    )

with col_result3:
    couverture_moyenne = latest_data['vacc_65_plus'].mean()
    st.metric(
        "Couverture moyenne actuelle",
        f"{couverture_moyenne:.1f}%",
        delta=f"{couverture_moyenne - objectif_couverture:.1f}% vs objectif"
    )

with col_result4:
    total_urgences = latest_data['urgences_grippe'].sum()
    st.metric(
        "Total urgences (semaine)",
        f"{total_urgences:,.0f}"
    )

st.markdown("")

# Graphique de priorité
st.markdown("### Scores de priorité par région")

fig_priority = px.bar(
    latest_data,
    x='region',
    y='score_priorite',
    color='categorie',
    color_discrete_map={
        'Critique': '#ef4444',
        'Élevée': '#f59e0b',
        'Moyenne': '#fbbf24',
        'Faible': '#22c55e'
    },
    labels={'score_priorite': 'Score de priorité', 'region': 'Région', 'categorie': 'Catégorie'},
    title="Régions classées par priorité d'allocation"
)

fig_priority.update_layout(
    height=400,
    xaxis_tickangle=-45,
    showlegend=True
)

st.plotly_chart(fig_priority, use_container_width=True)

# Graphique d'allocation
st.markdown("### Répartition des doses par région")

col_alloc1, col_alloc2 = st.columns(2)

with col_alloc1:
    fig_pie = px.pie(
        latest_data,
        values='doses_allouees',
        names='region',
        title="Distribution des doses (en %)",
        color_discrete_sequence=px.colors.sequential.RdBu_r
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_alloc2:
    fig_bar_doses = px.bar(
        latest_data.head(10),
        x='region',
        y='doses_allouees',
        color='doses_allouees',
        color_continuous_scale='Reds',
        labels={'doses_allouees': 'Doses allouées', 'region': 'Région'},
        title="Top 10 régions - Allocation de doses"
    )
    fig_bar_doses.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
    fig_bar_doses.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
    st.plotly_chart(fig_bar_doses, use_container_width=True)

# Analyse détaillée
st.markdown("### Analyse détaillée des critères")

# Graphique multi-critères
fig_multi = go.Figure()

fig_multi.add_trace(go.Bar(
    name='Déficit couverture',
    x=latest_data['region'],
    y=latest_data['deficit_norm'] * poids_deficit,
    marker_color='#3b82f6'
))

fig_multi.add_trace(go.Bar(
    name='Urgences',
    x=latest_data['region'],
    y=latest_data['urgences_norm'] * poids_urgences,
    marker_color='#ef4444'
))

fig_multi.add_trace(go.Bar(
    name='Tendance',
    x=latest_data['region'],
    y=latest_data['tendance_norm'] * poids_tendance,
    marker_color='#f59e0b'
))

fig_multi.update_layout(
    barmode='stack',
    title="Contribution de chaque critère au score de priorité",
    xaxis_title="Région",
    yaxis_title="Score pondéré",
    height=400,
    xaxis_tickangle=-45
)

st.plotly_chart(fig_multi, use_container_width=True)

# Tableau détaillé
st.markdown("### Plan de distribution détaillé")

display_data = latest_data[[
    'region', 'doses_allouees', 'score_priorite', 'categorie',
    'vacc_65_plus', 'deficit_couverture', 'urgences_grippe', 'tendance_urgences'
]].copy()

display_data.columns = [
    'Région', 'Doses allouées', 'Score priorité', 'Catégorie',
    'Couverture actuelle (%)', 'Déficit (%)', 'Urgences', 'Tendance urgences'
]

st.dataframe(
    display_data.style.background_gradient(
        subset=['Doses allouées'],
        cmap='Reds'
    ).background_gradient(
        subset=['Score priorité'],
        cmap='YlOrRd'
    ).background_gradient(
        subset=['Couverture actuelle (%)'],
        cmap='RdYlGn',
        vmin=30,
        vmax=80
    ).format({
        'Doses allouées': '{:,.0f}',
        'Score priorité': '{:.3f}',
        'Couverture actuelle (%)': '{:.1f}',
        'Déficit (%)': '{:.1f}',
        'Urgences': '{:.0f}',
        'Tendance urgences': '{:+.1f}'
    }),
    use_container_width=True,
    height=500
)

# Export
st.markdown("### Export des résultats")

col_export1, col_export2 = st.columns(2)

with col_export1:
    csv_data = display_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger le plan de distribution (CSV)",
        data=csv_data,
        file_name=f"plan_distribution_{latest_date.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col_export2:
    # Créer un résumé
    summary_text = f"""
PLAN DE DISTRIBUTION DES VACCINS
Date : {latest_date.strftime('%d/%m/%Y')}

PARAMÈTRES
- Doses disponibles : {doses_disponibles:,}
- Objectif couverture : {objectif_couverture}%
- Stratégie : {strategie}

RÉSULTATS
- Doses allouées : {latest_data['doses_allouees'].sum():,}
- Régions prioritaires : {nb_regions_prioritaires}
- Couverture moyenne : {couverture_moyenne:.1f}%

TOP 5 RÉGIONS PRIORITAIRES
{display_data.head(5)[['Région', 'Doses allouées', 'Catégorie']].to_string(index=False)}
    """

    st.download_button(
        label="Télécharger le résumé (TXT)",
        data=summary_text,
        file_name=f"resume_distribution_{latest_date.strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

# Recommandations
st.markdown("---")
st.markdown("### Recommandations")

col_rec1, col_rec2, col_rec3 = st.columns(3)

with col_rec1:
    regions_critiques = latest_data[latest_data['categorie'] == 'Critique']
    if len(regions_critiques) > 0:
        st.error(f"""
        **{len(regions_critiques)} région(s) en situation critique**

        Actions immédiates requises :
        - Allocation prioritaire
        - Renfort des stocks en pharmacie
        - Communication renforcée

        Régions : {', '.join(regions_critiques['region'].values)}
        """)
    else:
        st.success("Aucune région en situation critique")

with col_rec2:
    regions_hausse = latest_data[latest_data['tendance_urgences'] > 0]
    st.warning(f"""
    **{len(regions_hausse)} région(s) en hausse**

    Surveillance renforcée :
    - Monitoring hebdomadaire
    - Ajustement si nécessaire

    Tendance moyenne : {regions_hausse['tendance_urgences'].mean():+.1f} urgences/semaine
    """)

with col_rec3:
    deficit_moyen = latest_data['deficit_couverture'].mean()
    st.info(f"""
    **Déficit moyen : {deficit_moyen:.1f}%**

    Pour atteindre l'objectif national :
    - Campagne de sensibilisation
    - Faciliter l'accès en pharmacie
    - Communication ciblée
    """)

# Footer
st.markdown("---")
st.caption(f"Jonas - Optimisation de la distribution des vaccins | Calculé le {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
