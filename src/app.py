import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import requests
import json
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Jonas - Surveillance Grippale",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache pour les données GeoJSON
@st.cache_data
def load_geojson():
    """Charge les données GeoJSON de la France avec les DROM"""
    try:
        # URL du GeoJSON complet avec DROM
        url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-avec-outre-mer.geojson"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données géographiques: {e}")
        return None

@st.cache_data
def load_drom_geojson():
    """Extrait les géométries des DROM du GeoJSON complet"""
    full_geojson = load_geojson()
    if not full_geojson:
        return None
    
    # Codes départementaux des DROM
    drom_codes = {
        '971': 'Guadeloupe',
        '972': 'Martinique', 
        '973': 'Guyane',
        '974': 'Réunion',
        '976': 'Mayotte'
    }
    
    drom_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for feature in full_geojson['features']:
        code_dept = feature['properties'].get('code')
        if code_dept in drom_codes:
            # Ajouter la région correspondante
            feature['properties']['region'] = drom_codes[code_dept]
            drom_geojson['features'].append(feature)
    
    return drom_geojson

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

st.markdown("---")
st.markdown("---")
st.markdown("### Carte des urgences grippe par région")

last_week = data[data['date_semaine'] == latest_date].copy()

# Séparer métropole et DROM
metropole = last_week[~last_week['region'].isin(['Guadeloupe', 'Martinique', 'Guyane', 'Réunion', 'Mayotte'])]
drom = last_week[last_week['region'].isin(['Guadeloupe', 'Martinique', 'Guyane', 'Réunion', 'Mayotte'])]

# === CARTE FRANCE MÉTROPOLITAINE OPTIMISÉE ===
st.markdown("**France métropolitaine**")

# Optimisation: Réduire les données et simplifier le GeoJSON
geojson_url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson"

# Pre-filtrer les données pour éviter les calculs inutiles
metropole_clean = metropole.dropna(subset=['urgences_grippe']).copy()

fig_metro = px.choropleth(
    metropole_clean,
    geojson=geojson_url,
    locations='region',
    color='urgences_grippe',
    hover_name='region',
    hover_data={
        'urgences_grippe': ':,.0f',
        'vacc_65_plus': ':.1f'  # Retirer le % pour éviter les calculs
    },
    color_continuous_scale='Reds',
    labels={'urgences_grippe': 'Urgences grippe', 'vacc_65_plus': 'Couverture 65+ (%)'},
    featureidkey="properties.nom"
)

fig_metro.update_geos(
    fitbounds="locations",
    visible=False,
    resolution=110  # Réduire la résolution pour plus de rapidité
)

fig_metro.update_layout(
    height=400,  # Réduire la hauteur pour accélérer le rendu
    margin=dict(l=0, r=0, t=20, b=0),
    geo=dict(
        showframe=False,
        showcoastlines=False,  # Désactiver pour plus de rapidité
        projection_type='mercator'
    ),
    coloraxis_colorbar=dict(
        title=dict(
            text="Urgences grippe",
            side="right"
        ),
        len=0.8,
        thickness=12
    )
)

st.plotly_chart(fig_metro, use_container_width=True)

# === CARTES DROM AVEC GEOJSON ===
if len(drom) > 0:
    st.markdown("**Départements et régions d'outre-mer**")
    
    # Charger les données GeoJSON des DROM
    drom_geojson = load_drom_geojson()
    
    if drom_geojson:
        # Mapping des codes départementaux vers les régions
        dept_to_region = {
            '971': 'Guadeloupe',
            '972': 'Martinique', 
            '973': 'Guyane',
            '974': 'Réunion',
            '976': 'Mayotte'
        }
        
        # Configuration des vues pour chaque DROM
        drom_configs = {
            'Guadeloupe': {
                'center': {'lat': 16.25, 'lon': -61.58},
                'lonaxis_range': [-64, -59],
                'lataxis_range': [15, 18],
                'projection_scale': 12
            },
            'Martinique': {
                'center': {'lat': 14.64, 'lon': -61.02},
                'lonaxis_range': [-63, -58],
                'lataxis_range': [13, 16],
                'projection_scale': 12
            },
            'Guyane': {
                'center': {'lat': 4.0, 'lon': -53.0},
                'lonaxis_range': [-58, -48],
                'lataxis_range': [0, 8],
                'projection_scale': 4
            },
            'Réunion': {
                'center': {'lat': -21.13, 'lon': 55.52},
                'lonaxis_range': [53, 58],
                'lataxis_range': [-23, -19],
                'projection_scale': 12
            },
            'Mayotte': {
                'center': {'lat': -12.78, 'lon': 45.22},
                'lonaxis_range': [43, 47],
                'lataxis_range': [-14, -11],
                'projection_scale': 15
            }
        }
        
        # Organiser les DROM sur plusieurs lignes
        drom_regions = drom['region'].tolist()
        
        # Première ligne : 3 DROM
        if len(drom_regions) >= 3:
            col_drom1, col_drom2, col_drom3 = st.columns(3)
            cols_line1 = [col_drom1, col_drom2, col_drom3]
            
            for i, region in enumerate(drom_regions[:3]):
                if region in drom_configs:
                    with cols_line1[i]:
                        region_data = drom[drom['region'] == region]
                        config = drom_configs[region]
                        
                        # Métriques pour cette région avec gestion des NaN
                        urgences_val = region_data['urgences_grippe'].iloc[0]
                        vacc_val = region_data['vacc_65_plus'].iloc[0]
                        
                        st.markdown(f"**{region}**")
                        
                        # Métriques avec gestion des NaN
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            if pd.isna(urgences_val) or urgences_val == 0:
                                st.metric("Urgences", "N/A")
                            else:
                                st.metric("Urgences", f"{urgences_val:,.0f}")
                        with metric_col2:
                            if pd.isna(vacc_val):
                                st.metric("Vacc 65+", "N/A")
                            else:
                                st.metric("Vacc 65+", f"{vacc_val:.1f}%")
                        
                        # Créer une carte choroplèthe avec GeoJSON
                        if pd.isna(urgences_val) or urgences_val == 0:
                            # Carte simple sans données avec texte overlay
                            fig_drom = go.Figure()
                            
                            # Ajouter les contours du territoire
                            for feature in drom_geojson['features']:
                                if feature['properties'].get('region') == region:
                                    fig_drom.add_trace(go.Scattergeo(
                                        lon=[config['center']['lon']],
                                        lat=[config['center']['lat']],
                                        mode='markers',
                                        marker=dict(size=0, opacity=0),
                                        showlegend=False
                                    ))
                            
                            # Ajouter annotation de texte
                            fig_drom.add_annotation(
                                x=0.5, y=0.5,
                                xref="paper", yref="paper",
                                text="AUCUNE DONNÉE<br>D'URGENCES<br>DISPONIBLE",
                                showarrow=False,
                                font=dict(size=12, color="red", family="Arial"),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="red",
                                borderwidth=1,
                                xanchor="center",
                                yanchor="middle"
                            )
                        else:
                            # Préparer les données pour le choroplèthe
                            drom_dept_data = []
                            
                            # Trouver le code département correspondant à cette région
                            region_dept_code = None
                            for code, reg_name in dept_to_region.items():
                                if reg_name == region:
                                    region_dept_code = code
                                    break
                            
                            if region_dept_code:
                                drom_dept_data.append({
                                    'code': region_dept_code,
                                    'urgences_grippe': urgences_val,
                                    'region': region
                                })
                            
                            # Créer DataFrame pour le choroplèthe
                            df_drom_chart = pd.DataFrame(drom_dept_data)
                            
                            # Créer la carte choroplèthe
                            fig_drom = px.choropleth(
                                df_drom_chart,
                                geojson=drom_geojson,
                                locations='code',
                                featureidkey="properties.code",
                                color='urgences_grippe',
                                hover_name='region',
                                hover_data={'urgences_grippe': ':,.0f', 'code': False},
                                color_continuous_scale='Reds',
                                range_color=[0, df_drom_chart['urgences_grippe'].max()] if not df_drom_chart.empty else [0, 1]
                            )
                        
                        # Configuration de la vue géographique
                        fig_drom.update_geos(
                            center=config['center'],
                            lonaxis_range=config['lonaxis_range'],
                            lataxis_range=config['lataxis_range'],
                            projection_scale=config['projection_scale'],
                            showframe=False,
                            showcoastlines=True,
                            projection_type='mercator',
                            oceancolor="lightblue",
                            coastlinewidth=1,
                            coastlinecolor="gray"
                        )
                        
                        fig_drom.update_layout(
                            height=200,
                            margin=dict(l=0, r=0, t=5, b=0),
                            showlegend=False,
                            coloraxis_showscale=False
                        )
                        
                        st.plotly_chart(fig_drom, use_container_width=True, key=f"drom_{region}_geojson_line1")
        
        # Deuxième ligne : DROM restants (s'il y en a)
        if len(drom_regions) > 3:
            remaining_regions = drom_regions[3:]
            
            if len(remaining_regions) == 1:
                col_center = st.columns([1, 1, 1])[1]
                cols_line2 = [col_center]
            elif len(remaining_regions) == 2:
                col_left, col_right = st.columns([1, 1])
                cols_line2 = [col_left, col_right]
            else:
                col_d1, col_d2, col_d3 = st.columns(3)
                cols_line2 = [col_d1, col_d2, col_d3]
            
            for i, region in enumerate(remaining_regions[:3]):
                if region in drom_configs and i < len(cols_line2):
                    with cols_line2[i]:
                        region_data = drom[drom['region'] == region]
                        config = drom_configs[region]
                        
                        # Métriques pour cette région avec gestion des NaN
                        urgences_val = region_data['urgences_grippe'].iloc[0]
                        vacc_val = region_data['vacc_65_plus'].iloc[0]
                        
                        st.markdown(f"**{region}**")
                        
                        # Métriques avec gestion des NaN
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            if pd.isna(urgences_val) or urgences_val == 0:
                                st.metric("Urgences", "N/A")
                            else:
                                st.metric("Urgences", f"{urgences_val:,.0f}")
                        with metric_col2:
                            if pd.isna(vacc_val):
                                st.metric("Vacc 65+", "N/A")
                            else:
                                st.metric("Vacc 65+", f"{vacc_val:.1f}%")
                        
                        # Créer une carte choroplèthe avec GeoJSON (même logique que ligne 1)
                        if pd.isna(urgences_val) or urgences_val == 0:
                            fig_drom = go.Figure()
                            
                            fig_drom.add_trace(go.Scattergeo(
                                lon=[config['center']['lon']],
                                lat=[config['center']['lat']],
                                mode='markers',
                                marker=dict(size=0, opacity=0),
                                showlegend=False
                            ))
                            
                            fig_drom.add_annotation(
                                x=0.5, y=0.5,
                                xref="paper", yref="paper",
                                text="AUCUNE DONNÉE<br>D'URGENCES<br>DISPONIBLE",
                                showarrow=False,
                                font=dict(size=12, color="red", family="Arial"),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="red",
                                borderwidth=1,
                                xanchor="center",
                                yanchor="middle"
                            )
                        else:
                            # Préparer les données pour le choroplèthe
                            drom_dept_data = []
                            region_dept_code = None
                            for code, reg_name in dept_to_region.items():
                                if reg_name == region:
                                    region_dept_code = code
                                    break
                            
                            if region_dept_code:
                                drom_dept_data.append({
                                    'code': region_dept_code,
                                    'urgences_grippe': urgences_val,
                                    'region': region
                                })
                            
                            df_drom_chart = pd.DataFrame(drom_dept_data)
                            
                            fig_drom = px.choropleth(
                                df_drom_chart,
                                geojson=drom_geojson,
                                locations='code',
                                featureidkey="properties.code",
                                color='urgences_grippe',
                                hover_name='region',
                                hover_data={'urgences_grippe': ':,.0f', 'code': False},
                                color_continuous_scale='Reds',
                                range_color=[0, df_drom_chart['urgences_grippe'].max()] if not df_drom_chart.empty else [0, 1]
                            )
                        
                        fig_drom.update_geos(
                            center=config['center'],
                            lonaxis_range=config['lonaxis_range'],
                            lataxis_range=config['lataxis_range'],
                            projection_scale=config['projection_scale'],
                            showframe=False,
                            showcoastlines=True,
                            projection_type='mercator',
                            oceancolor="lightblue",
                            coastlinewidth=1,
                            coastlinecolor="gray"
                        )
                        
                        fig_drom.update_layout(
                            height=200,
                            margin=dict(l=0, r=0, t=5, b=0),
                            showlegend=False,
                            coloraxis_showscale=False
                        )
                        
                        st.plotly_chart(fig_drom, use_container_width=True, key=f"drom_{region}_geojson_line2")
    else:
        st.error("Impossible de charger les données géographiques des DROM")

else:
    st.info("Aucune donnée DROM disponible")


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
