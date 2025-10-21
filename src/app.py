import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Jonas - Surveillance Grippale",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f9fafb;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/master_dataframe.csv')
    df['date_semaine'] = pd.to_datetime(df['date_semaine'])
    return df

data = load_data()

# Calculer les données nationales (agrégation de toutes les régions)
national_data = data.groupby('date_semaine').agg({
    'urgences_grippe': 'sum',
    'sos_medecins': 'sum',
    'vacc_65_plus': 'mean',
    'vacc_moins_65_risque': 'mean',
    'vacc_65_74': 'mean',
    'vacc_75_plus': 'mean'
}).reset_index()

national_data = national_data.sort_values('date_semaine')

# Données récentes
recent_data = national_data.tail(12).copy()
latest_date = national_data['date_semaine'].max()

# Métriques actuelles vs semaine précédente
current_urgences = recent_data['urgences_grippe'].iloc[-1]
previous_urgences = recent_data['urgences_grippe'].iloc[-2]
urgences_delta = ((current_urgences - previous_urgences) / previous_urgences * 100) if previous_urgences > 0 else 0

current_sos = recent_data['sos_medecins'].iloc[-1]
previous_sos = recent_data['sos_medecins'].iloc[-2]
sos_delta = ((current_sos - previous_sos) / previous_sos * 100) if previous_sos > 0 else 0

current_vacc_65 = recent_data['vacc_65_plus'].iloc[-1]
current_vacc_risque = recent_data['vacc_moins_65_risque'].iloc[-1]

# Calcul des tendances (4 dernières semaines)
last_4_weeks = recent_data.tail(4)
trend_urgences = last_4_weeks['urgences_grippe'].diff().mean()

# Déterminer le niveau d'alerte
if trend_urgences > 1000:
    alert_level = "error"
    alert_text = "Forte hausse"
elif trend_urgences > 0:
    alert_level = "warning"
    alert_text = "Hausse modérée"
elif trend_urgences < -1000:
    alert_level = "success"
    alert_text = "Baisse significative"
else:
    alert_level = "info"
    alert_text = "Situation stable"

# Header
st.title("Tableau de bord Jonas")
st.markdown("**Surveillance de l'activité grippale en France**")
st.markdown("")

# Info dernière mise à jour
col_info1, col_info2 = st.columns([2, 1])
with col_info1:
    st.info(f"Dernières données : Semaine du {latest_date.strftime('%d/%m/%Y')}")
with col_info2:
    if alert_level == "error":
        st.error(f"{alert_text}")
    elif alert_level == "warning":
        st.warning(f"{alert_text}")
    elif alert_level == "success":
        st.success(f"{alert_text}")
    else:
        st.info(f"{alert_emoji} {alert_text}")

st.markdown("")

# KPI Cards - Ligne 1: Urgences & SOS
st.markdown("### Activité hospitalière")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Urgences grippe (semaine)",
        value=f"{current_urgences:,.0f}",
        delta=f"{urgences_delta:+.1f}%",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="SOS Médecins (semaine)",
        value=f"{current_sos:,.0f}",
        delta=f"{sos_delta:+.1f}%",
        delta_color="inverse"
    )

with col3:
    # Somme des 4 dernières semaines
    total_4_weeks_urgences = last_4_weeks['urgences_grippe'].sum()
    st.metric(
        label="Total 4 dernières semaines",
        value=f"{total_4_weeks_urgences:,.0f}",
        delta="Urgences"
    )

with col4:
    total_4_weeks_sos = last_4_weeks['sos_medecins'].sum()
    st.metric(
        label="Total 4 dernières semaines",
        value=f"{total_4_weeks_sos:,.0f}",
        delta="SOS Médecins"
    )


st.markdown("---")

# Graphique principal - Évolution des urgences et SOS
st.markdown("### Évolution de l'activité grippale (3 dernières années)")

# Récupérer les données des 3 dernières années
max_year = national_data['date_semaine'].dt.year.max()
min_year = max_year - 2
three_years_data = national_data[national_data['date_semaine'].dt.year >= min_year].copy()

# Supprimer les valeurs nulles pour éviter les bugs
three_years_data = three_years_data.dropna(subset=['urgences_grippe', 'sos_medecins'])

fig_main = go.Figure()

# Urgences
fig_main.add_trace(go.Scatter(
    x=three_years_data['date_semaine'],
    y=three_years_data['urgences_grippe'],
    mode='lines',
    name='Passages aux urgences',
    line=dict(color='#ef4444', width=2),
    yaxis='y'
))

# SOS Médecins
fig_main.add_trace(go.Scatter(
    x=three_years_data['date_semaine'],
    y=three_years_data['sos_medecins'],
    mode='lines',
    name='Interventions SOS Médecins',
    line=dict(color='#3b82f6', width=2),
    yaxis='y2'
))

fig_main.update_layout(
    height=450,
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        title="Date",
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis=dict(
        title=dict(text="Passages aux urgences", font=dict(color="#ef4444")),
        tickfont=dict(color="#ef4444")
    ),
    yaxis2=dict(
        title=dict(text="Interventions SOS Médecins", font=dict(color="#3b82f6")),
        tickfont=dict(color="#3b82f6"),
        overlaying='y',
        side='right'
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)


st.plotly_chart(fig_main, use_container_width=True)

st.markdown("")

# Graphiques secondaires
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("### Top 5 régions - Urgences grippe")

    last_week = data[data['date_semaine'] == latest_date]
    top_regions = last_week.nlargest(5, 'urgences_grippe')[['region', 'urgences_grippe']]

    fig_regions = px.bar(
        top_regions,
        x='region',
        y='urgences_grippe',
        color='urgences_grippe',
        color_continuous_scale='Reds',
        labels={'urgences_grippe': 'Passages aux urgences', 'region': 'Région'}
    )

    fig_regions.update_layout(
        height=350,
        showlegend=False,
        xaxis_tickangle=-45
    )
    fig_regions.update_traces(texttemplate='%{y:,.0f}', textposition='outside')

    st.plotly_chart(fig_regions, use_container_width=True)

with col_chart2:
    st.markdown("### Vaccination par région (65+)")

    vacc_by_region = last_week[['region', 'vacc_65_plus']].dropna().sort_values('vacc_65_plus', ascending=False)

    fig_vacc = px.bar(
        vacc_by_region,
        x='region',
        y='vacc_65_plus',
        color='vacc_65_plus',
        color_continuous_scale='RdYlGn',
        range_color=[30, 80],
        labels={'vacc_65_plus': 'Couverture (%)', 'region': 'Région'}
    )

    fig_vacc.update_layout(
        height=350,
        showlegend=False,
        xaxis_tickangle=-45
    )
    fig_vacc.update_traces(texttemplate='%{y:.1f}%', textposition='outside')

    st.plotly_chart(fig_vacc, use_container_width=True)

st.markdown("")


st.markdown("---")

# Statistiques nationales détaillées
st.markdown("### Statistiques nationales (dernière semaine)")

stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    avg_vacc_65 = last_week['vacc_65_plus'].mean()
    st.metric("Couverture 65+ moy.", f"{avg_vacc_65:.1f}%")

with stat_col2:
    nb_regions = len(last_week)
    st.metric("Régions suivies", f"{nb_regions}")

with stat_col3:
    # Régions avec faible couverture
    low_coverage = len(last_week[last_week['vacc_65_plus'] < 50])
    st.metric("Régions < 50% vacc.", f"{low_coverage}", delta_color="inverse")

st.markdown("")
st.markdown("")


# Tableau récapitulatif par région
st.markdown("### Tableau récapitulatif par région (dernière semaine)")

summary_table = last_week[['region', 'urgences_grippe', 'sos_medecins', 'vacc_65_plus', 'vacc_moins_65_risque']].copy()
summary_table.columns = ['Région', 'Urgences', 'SOS Médecins', 'Vacc 65+', 'Vacc <65 risque']
summary_table = summary_table.sort_values('Urgences', ascending=False)

st.dataframe(
    summary_table.style.background_gradient(
        subset=['Urgences', 'SOS Médecins'],
        cmap='Reds',
        vmin=0
    ).background_gradient(
        subset=['Vacc 65+', 'Vacc <65 risque'],
        cmap='RdYlGn',
        vmin=30,
        vmax=80
    ).format({
        'Urgences': '{:,.0f}',
        'SOS Médecins': '{:,.0f}',
        'Vacc 65+': '{:.1f}%',
        'Vacc <65 risque': '{:.1f}%'
    }),
    use_container_width=True,
    height=400
)

# Footer
st.markdown("---")
st.caption(f"Jonas - Surveillance de l'activité grippale en France | Dernières données : {latest_date.strftime('%d/%m/%Y')} | {nb_regions} régions")
