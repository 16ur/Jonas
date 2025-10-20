import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

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
    .alert-card {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .stAlert {
        border-radius: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Données
historicalData = pd.DataFrame({
    'week': ["S40", "S41", "S42", "S43", "S44", "S45", "S46", "S47", "S48", "S49", "S50", "S51", "S52"],
    'ias': [45, 52, 61, 75, 89, 105, 128, 156, None, None, None, None, None],
    'erVisits': [120, 145, 178, 220, 285, 340, 420, 510, None, None, None, None, None],
    'predicted': [None, None, None, None, None, None, None, 515, 625, 745, 850, 920, 880]
})

dailyIAS = pd.DataFrame({
    'day': ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
    'value': [148, 152, 156, 159, 162, 165, 168]
})

# Variables
currentIAS = 156
predictedPeak = 920
weeksUntilPeak = 4
alertLevel = "warning"

# Header
st.title("📊 Tableau de bord Jonas")
st.markdown("**Suivi en temps réel de l'activité grippale et prévisions**")
st.markdown("")

# Bannière d'alerte
if alertLevel == "warning":
    st.warning(f"""
    ### ⚠️ Alerte : Pic épidémique prévu
    
    Un pic d'activité grippale est prévu dans **{weeksUntilPeak} semaines** avec 
    environ **{predictedPeak} passages aux urgences** prévus.
    """)
    
    if st.button("🔍 Voir les détails", key="alert_details"):
        st.info("Redirection vers le modèle prédictif...")

st.markdown("")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📍 IAS® actuel",
        value=currentIAS,
        delta="+8%",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="📈 Pic prévu (passages)",
        value=predictedPeak,
        delta="+125%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="📅 Semaines avant pic",
        value=weeksUntilPeak,
        delta=None
    )

with col4:
    st.metric(
        label="🎯 Précision modèle",
        value="85%",
        delta=None
    )

st.markdown("")

# Graphiques - Ligne 1
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("#### 📊 Évolution IAS® (7 derniers jours)")
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=dailyIAS['day'],
        y=dailyIAS['value'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#2563eb', width=3),
        fillcolor='rgba(37, 99, 235, 0.1)',
        name='IAS®'
    ))
    
    fig_daily.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="",
        yaxis_title="IAS®",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_daily, use_container_width=True)

with col_chart2:
    st.markdown("#### 🚨 Statut des alertes")
    
    st.markdown("""
        <div style='background: #fff7ed; border: 2px solid #fdba74; border-radius: 0.75rem; padding: 1rem; margin-bottom: 0.5rem;'>
            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                <div style='width: 12px; height: 12px; background: #f97316; border-radius: 50%; animation: pulse 2s infinite;'></div>
                <div>
                    <div style='font-weight: 500; margin-bottom: 0.25rem;'>⚠️ Alerte pic épidémique</div>
                    <div style='font-size: 0.75rem; color: #78716c;'>Activée il y a 2 jours</div>
                </div>
            </div>
        </div>
        
        <div style='background: #f0fdf4; border: 2px solid #86efac; border-radius: 0.75rem; padding: 1rem; margin-bottom: 0.5rem;'>
            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                <div style='width: 12px; height: 12px; background: #22c55e; border-radius: 50%;'></div>
                <div>
                    <div style='font-weight: 500; margin-bottom: 0.25rem;'>✅ Capacité normale</div>
                    <div style='font-size: 0.75rem; color: #78716c;'>Aucune saturation</div>
                </div>
            </div>
        </div>
        
        <div style='background: #eff6ff; border: 2px solid #93c5fd; border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem;'>
            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                <div style='width: 12px; height: 12px; background: #3b82f6; border-radius: 50%;'></div>
                <div>
                    <div style='font-weight: 500; margin-bottom: 0.25rem;'>ℹ️ Surveillance renforcée</div>
                    <div style='font-size: 0.75rem; color: #78716c;'>Activité en hausse</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔧 Gérer les alertes", use_container_width=True):
        st.info("Redirection vers la gestion des alertes...")

st.markdown("")

# Graphique principal - Corrélation et prévisions
st.markdown("#### 📉 Corrélation IAS® & Passages aux urgences")
st.caption("Données historiques et prévisions pour les 5 prochaines semaines")

fig_main = go.Figure()

# Ligne IAS®
fig_main.add_trace(go.Scatter(
    x=historicalData['week'],
    y=historicalData['ias'],
    mode='lines+markers',
    name='IAS®',
    line=dict(color='#2563eb', width=3),
    marker=dict(size=8, color='#2563eb')
))

# Ligne Passages urgences
fig_main.add_trace(go.Scatter(
    x=historicalData['week'],
    y=historicalData['erVisits'],
    mode='lines+markers',
    name='Passages urgences',
    line=dict(color='#06b6d4', width=3),
    marker=dict(size=8, color='#06b6d4')
))

# Ligne Prévisions
fig_main.add_trace(go.Scatter(
    x=historicalData['week'],
    y=historicalData['predicted'],
    mode='lines+markers',
    name='Prévision',
    line=dict(color='#f59e0b', width=3, dash='dash'),
    marker=dict(size=8, color='#f59e0b')
))

# Ligne verticale "Aujourd'hui"
fig_main.add_vline(
    x="S47",
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="Aujourd'hui",
    annotation_position="top"
)

fig_main.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Semaine",
    yaxis_title="Valeur",
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_main, use_container_width=True)

# Légendes
col_leg1, col_leg2, col_leg3 = st.columns(3)

with col_leg1:
    st.info("""
    **🔵 IAS®**  
    Indicateur d'activité de la surveillance grippale
    """)

with col_leg2:
    st.info("""
    **🔷 Passages urgences**  
    Nombre de passages aux urgences pour grippe
    """)

with col_leg3:
    st.info("""
    **🟠 Prévision**  
    Estimation basée sur le modèle prédictif (85% précision)
    """)

# Footer
st.markdown("---")
st.caption("Jonas - Système de surveillance et prédiction de l'activité grippale | Dernière mise à jour : " + datetime.now().strftime("%d/%m/%Y %H:%M"))