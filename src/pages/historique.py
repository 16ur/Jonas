"""
Page : Données Historiques
Analyse des saisons épidémiques et corrélations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ===== CONFIGURATION PAGE =====
st.set_page_config(
    page_title="Données Historiques - GrippeAlert",
    page_icon="📅",
    layout="wide"
)

# ===== DONNÉES MOCK (à remplacer par tes vraies données) =====

# Données saisonnières
seasonal_data = pd.DataFrame({
    'week': ["S40", "S42", "S44", "S46", "S48", "S50", "S52", "S02", "S04", "S06", "S08"],
    'season2022': [45, 62, 88, 125, 178, 245, 312, 285, 198, 125, 75],
    'season2023': [52, 71, 98, 142, 195, 268, 338, 305, 215, 142, 85],
    'season2024': [61, 89, 128, 156, None, None, None, None, None, None, None]
})

# Données de corrélation
correlation_data = pd.DataFrame({
    'ias': [45, 52, 61, 75, 89, 105, 128, 156, 178, 245, 312],
    'urgences': [120, 145, 178, 220, 285, 340, 420, 510, 585, 725, 920]
})

# Données pics épidémiques
peaks_data = pd.DataFrame({
    'saison': ['2021-22', '2022-23', '2023-24', '2024-25'],
    'pic': [298, 312, 338, 156],
    'semaine': ['S01', 'S52', 'S52', 'S46']
})


# ===== HEADER =====
st.title("📅 Données Historiques")
st.markdown("Analyse des saisons épidémiques précédentes et corrélations IAS®")
st.markdown("---")


# ===== MÉTRIQUES PRINCIPALES =====
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🗓️ Saisons analysées",
        value="3",
        help="Nombre de saisons épidémiques dans l'analyse"
    )

with col2:
    st.metric(
        label="📈 Corrélation R²",
        value="0.89",
        delta="Forte corrélation",
        help="Coefficient de corrélation entre IAS® et urgences"
    )

with col3:
    st.metric(
        label="🔥 IAS® max (2022)",
        value="312",
        help="Valeur maximale d'IAS® enregistrée"
    )

st.markdown("<br>", unsafe_allow_html=True)


# ===== GRAPHIQUE 1 : COMPARAISON SAISONNIÈRE =====
st.markdown("### 📊 Comparaison des saisons épidémiques")
st.caption("Évolution de l'IAS® au cours des 3 dernières saisons")

# Bouton export
col_title, col_export = st.columns([3, 1])
with col_export:
    if st.button("⬇️ Exporter données", key="export_seasonal"):
        csv = seasonal_data.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="donnees_saisonnieres.csv",
            mime="text/csv"
        )

# Graphique
fig_seasonal = go.Figure()

# Saison 2022-2023
fig_seasonal.add_trace(go.Scatter(
    x=seasonal_data['week'],
    y=seasonal_data['season2022'],
    mode='lines+markers',
    name='Saison 2022-2023',
    line=dict(color='#94a3b8', width=2),
    marker=dict(size=8, color='#94a3b8')
))

# Saison 2023-2024
fig_seasonal.add_trace(go.Scatter(
    x=seasonal_data['week'],
    y=seasonal_data['season2023'],
    mode='lines+markers',
    name='Saison 2023-2024',
    line=dict(color='#06b6d4', width=2),
    marker=dict(size=8, color='#06b6d4')
))

# Saison 2024-2025 (actuelle)
fig_seasonal.add_trace(go.Scatter(
    x=seasonal_data['week'],
    y=seasonal_data['season2024'],
    mode='lines+markers',
    name='Saison 2024-2025 (actuelle)',
    line=dict(color='#2563eb', width=3),
    marker=dict(size=10, color='#2563eb')
))

fig_seasonal.update_layout(
    xaxis_title="Semaine",
    yaxis_title="IAS®",
    hovermode='x unified',
    height=450,
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_seasonal, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# ===== SECTION 2 COLONNES =====
col_left, col_right = st.columns(2)

# ===== COLONNE GAUCHE : CORRÉLATION =====
with col_left:
    st.markdown("### 🔗 Corrélation IAS® & Passages aux urgences")
    
    # Scatter plot
    fig_corr = go.Figure()
    
    fig_corr.add_trace(go.Scatter(
        x=correlation_data['ias'],
        y=correlation_data['urgences'],
        mode='markers',
        name='Données',
        marker=dict(
            size=12,
            color='#2563eb',
            opacity=0.7,
            line=dict(color='white', width=2)
        )
    ))
    
    # Ligne de régression
    import numpy as np
    z = np.polyfit(correlation_data['ias'], correlation_data['urgences'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(correlation_data['ias'].min(), correlation_data['ias'].max(), 100)
    
    fig_corr.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name='Régression',
        line=dict(color='#ef4444', width=2, dash='dash')
    ))
    
    fig_corr.update_layout(
        xaxis_title="IAS®",
        yaxis_title="Passages aux urgences",
        height=400,
        template="plotly_white",
        showlegend=True
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Résultat corrélation
    st.info(f"""
    **Coefficient de corrélation (R²) : 0.89**
    
    Forte corrélation positive entre l'IAS® et les passages aux urgences.
    Plus l'IAS® augmente, plus les passages aux urgences augmentent.
    """)


# ===== COLONNE DROITE : PICS ÉPIDÉMIQUES =====
with col_right:
    st.markdown("### 🔥 Pics épidémiques par saison")
    
    # Bar chart
    fig_peaks = go.Figure()
    
    fig_peaks.add_trace(go.Bar(
        x=peaks_data['saison'],
        y=peaks_data['pic'],
        text=peaks_data['pic'],
        textposition='outside',
        marker=dict(
            color='#2563eb',
            cornerradius=8
        ),
        hovertemplate='<b>%{x}</b><br>Pic: %{y} IAS®<extra></extra>'
    ))
    
    fig_peaks.update_layout(
        xaxis_title="Saison",
        yaxis_title="IAS® au pic",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    st.plotly_chart(fig_peaks, use_container_width=True)
    
    # Statistiques
    st.markdown("**📊 Statistiques clés**")
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Pic moyen", "316 IAS®")
        st.metric("Durée moyenne épidémie", "12-14 semaines")
    
    with metric_col2:
        st.metric("Semaine type du pic", "S52 - S02")
        st.metric("Variation 2023-2024", "+8.3%")

st.markdown("<br>", unsafe_allow_html=True)


# ===== TABLEAU DE DONNÉES DÉTAILLÉ =====
st.markdown("### 📋 Tableau de données détaillé")

# Préparer données pour tableau avec variation
seasonal_display = seasonal_data.copy()
seasonal_display['variation'] = seasonal_display.apply(
    lambda row: f"+{((row['season2024'] - row['season2023']) / row['season2023'] * 100):.1f}%" 
    if pd.notna(row['season2024']) and pd.notna(row['season2023']) 
    else "—",
    axis=1
)

# Renommer colonnes
seasonal_display.columns = ['Semaine', 'IAS® 2022', 'IAS® 2023', 'IAS® 2024', 'Variation']

# Remplacer NaN par "—"
seasonal_display = seasonal_display.fillna("—")

# Afficher tableau
st.dataframe(
    seasonal_display,
    use_container_width=True,
    height=350,
    hide_index=True
)

# Bouton export tableau
col_export1, col_export2, col_export3 = st.columns([2, 1, 2])
with col_export2:
    csv_full = seasonal_display.to_csv(index=False)
    st.download_button(
        label="⬇️ Exporter CSV complet",
        data=csv_full,
        file_name="donnees_historiques_completes.csv",
        mime="text/csv",
        use_container_width=True
    )


# ===== INSIGHTS CLÉS =====
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 💡 Insights clés")

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.success("""
    **🎯 Prédictibilité**
    
    Les pics épidémiques surviennent systématiquement entre les semaines S50 et S02 (fin décembre - début janvier).
    """)

with insight_col2:
    st.info("""
    **📈 Tendance 2024**
    
    La saison 2024-2025 montre une augmentation de +26% par rapport à 2023 à semaine équivalente (S46).
    """)

with insight_col3:
    st.warning("""
    **⚠️ Alerte précoce**
    
    L'IAS® permet de détecter une montée épidémique 2-3 semaines avant le pic des urgences.
    """)


# ===== FOOTER =====
st.markdown("---")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.caption("Données : Santé Publique France | Modèle : GrippeAlert v1.0")

