"""
Page : Modèle Prédictif
Analyse des prévisions et performance du modèle ML
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# ===== CONFIGURATION PAGE =====
st.set_page_config(
    page_title="Modèle Prédictif - Jonas",
    page_icon="🤖",
    layout="wide"
)

# ===== DONNÉES MOCK =====

# Données de prédiction avec intervalles de confiance
prediction_data = pd.DataFrame({
    'week': ["S44", "S45", "S46", "S47", "S48", "S49", "S50", "S51", "S52", "S01", "S02"],
    'actual': [128, 142, 156, None, None, None, None, None, None, None, None],
    'predicted': [None, None, None, 185, 225, 285, 365, 465, 575, 685, 765],
    'lower': [None, None, None, 165, 200, 255, 330, 420, 520, 620, 695],
    'upper': [None, None, None, 205, 250, 315, 400, 510, 630, 750, 835]
})

# Évolution de la précision
accuracy_data = pd.DataFrame({
    'week': ["S40", "S41", "S42", "S43", "S44", "S45", "S46"],
    'accuracy': [78, 81, 83, 84, 85, 86, 85]
})


# ===== HEADER =====
st.title("🤖 Modèle Prédictif")
st.markdown("Analyse des prévisions et performance du modèle de machine learning")
st.markdown("---")


# ===== MÉTRIQUES PRINCIPALES =====
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #EEF2FF 0%, #DBEAFE 100%); border-radius: 16px;'>
        <div style='font-size: 32px; font-weight: bold; margin-bottom: 8px;'>🧠</div>
        <div style='font-size: 24px; font-weight: bold; margin-bottom: 4px;'>Random Forest</div>
        <div style='font-size: 14px; color: #6B7280;'>Type de modèle</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border-radius: 16px;'>
        <div style='font-size: 32px; font-weight: bold; margin-bottom: 8px;'>🎯</div>
        <div style='font-size: 24px; font-weight: bold; margin-bottom: 4px;'>89%</div>
        <div style='font-size: 14px; color: #6B7280;'>Précision (R²)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%); border-radius: 16px;'>
        <div style='font-size: 32px; font-weight: bold; margin-bottom: 8px;'>📅</div>
        <div style='font-size: 24px; font-weight: bold; margin-bottom: 4px;'>3 ans</div>
        <div style='font-size: 14px; color: #6B7280;'>Données entraînement</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%); border-radius: 16px;'>
        <div style='font-size: 32px; font-weight: bold; margin-bottom: 8px;'>📈</div>
        <div style='font-size: 24px; font-weight: bold; margin-bottom: 4px;'>2-3</div>
        <div style='font-size: 14px; color: #6B7280;'>Semaines d'avance</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ===== BANNER INFORMATIF =====
st.info("""
### 💡 Comment fonctionne le modèle ?

Notre modèle utilise un algorithme **Random Forest** entraîné sur 3 années de données de vaccination 
et de passages aux urgences. Il analyse les tendances temporelles et les corrélations pour prédire 
l'évolution de l'épidémie avec une précision de **89% (R²)**.

Les **intervalles de confiance** (zones grisées) représentent la marge d'erreur du modèle, 
permettant une interprétation prudente des prévisions. Plus l'intervalle est large, plus l'incertitude est grande.

**Variables utilisées :** Vaccinations hebdomadaires, passages urgences historiques, saisonnalité, tendance, moyennes mobiles.
""")

st.markdown("<br>", unsafe_allow_html=True)


# ===== GRAPHIQUE PRINCIPAL : PRÉDICTIONS =====
st.markdown("### 🔮 Prévisions avec intervalles de confiance")
st.caption("Prévisions de passages aux urgences pour les 8 prochaines semaines avec marges d'erreur")

# Créer le graphique
fig_pred = go.Figure()

# Intervalle de confiance (zone grisée)
fig_pred.add_trace(go.Scatter(
    x=prediction_data['week'],
    y=prediction_data['upper'],
    fill=None,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig_pred.add_trace(go.Scatter(
    x=prediction_data['week'],
    y=prediction_data['lower'],
    fill='tonexty',
    mode='lines',
    line=dict(width=0),
    name='Intervalle de confiance',
    fillcolor='rgba(148, 163, 184, 0.2)',
    hovertemplate='<b>%{x}</b><br>IC: %{y}<extra></extra>'
))

# Valeurs réelles
fig_pred.add_trace(go.Scatter(
    x=prediction_data['week'],
    y=prediction_data['actual'],
    mode='lines+markers',
    name='Valeurs réelles',
    line=dict(color='#2563eb', width=3),
    marker=dict(size=10, color='#2563eb'),
    hovertemplate='<b>%{x}</b><br>Réel: %{y}<extra></extra>'
))

# Prédictions
fig_pred.add_trace(go.Scatter(
    x=prediction_data['week'],
    y=prediction_data['predicted'],
    mode='lines+markers',
    name='Prévisions',
    line=dict(color='#f59e0b', width=3, dash='dash'),
    marker=dict(size=10, color='#f59e0b'),
    hovertemplate='<b>%{x}</b><br>Prévu: %{y}<extra></extra>'
))

# Ligne "Aujourd'hui"
fig_pred.add_vline(
    x=2,  # Index de S46
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="Aujourd'hui",
    annotation_position="top"
)

fig_pred.update_layout(
    xaxis_title="Semaine",
    yaxis_title="Passages aux urgences (pour 100k hab)",
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

st.plotly_chart(fig_pred, use_container_width=True)

# Métriques prédictions
pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #EEF2FF 0%, #DBEAFE 100%); border-radius: 12px;'>
        <div style='font-size: 14px; margin-bottom: 8px;'>Pic prévu</div>
        <div style='font-size: 28px; font-weight: bold; margin-bottom: 4px;'>765 passages</div>
        <div style='font-size: 12px; color: #6B7280;'>Semaine S02 (début janvier)</div>
    </div>
    """, unsafe_allow_html=True)

with pred_col2:
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%); border-radius: 12px;'>
        <div style='font-size: 14px; margin-bottom: 8px;'>Intervalle de confiance</div>
        <div style='font-size: 28px; font-weight: bold; margin-bottom: 4px;'>695 - 835</div>
        <div style='font-size: 12px; color: #6B7280;'>Marge d'erreur ±9%</div>
    </div>
    """, unsafe_allow_html=True)

with pred_col3:
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border-radius: 12px;'>
        <div style='font-size: 14px; margin-bottom: 8px;'>Délai d'anticipation</div>
        <div style='font-size: 28px; font-weight: bold; margin-bottom: 4px;'>8 semaines</div>
        <div style='font-size: 12px; color: #6B7280;'>Jusqu'au pic prévu</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# ===== SECTION 2 COLONNES =====
col_left, col_right = st.columns(2)

# ===== COLONNE GAUCHE : PRÉCISION DU MODÈLE =====
with col_left:
    st.markdown("### 📊 Évolution de la précision du modèle")
    
    # Graphique précision
    fig_acc = go.Figure()
    
    fig_acc.add_trace(go.Scatter(
        x=accuracy_data['week'],
        y=accuracy_data['accuracy'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.2)',
        line=dict(color='#10b981', width=3),
        name='Précision',
        hovertemplate='<b>%{x}</b><br>Précision: %{y}%<extra></extra>'
    ))
    
    fig_acc.update_layout(
        xaxis_title="Semaine",
        yaxis_title="Précision (%)",
        yaxis=dict(range=[70, 90]),
        height=350,
        template="plotly_white",
        showlegend=False
    )
    
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Métrique précision moyenne
    avg_accuracy = accuracy_data['accuracy'].mean()
    st.success(f"""
    **Précision moyenne (7 derniers jours) : {avg_accuracy:.1f}%**
    
    Le modèle maintient une précision stable et élevée, 
    garantissant des prédictions fiables.
    """)


# ===== COLONNE DROITE : CARACTÉRISTIQUES =====
with col_right:
    st.markdown("### ⚙️ Caractéristiques du modèle")
    
    # Caractéristiques
    st.markdown("""
    <div style='padding: 16px; background: #F9FAFB; border-radius: 12px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='font-weight: 500;'>Architecture</span>
            <span style='color: #2563eb;'>Random Forest</span>
        </div>
        <div style='font-size: 13px; color: #6B7280;'>
            Ensemble de 100 arbres de décision pour robustesse
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='padding: 16px; background: #F9FAFB; border-radius: 12px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='font-weight: 500;'>Variables d'entrée</span>
            <span style='color: #2563eb;'>6 features</span>
        </div>
        <div style='font-size: 13px; color: #6B7280;'>
            Vaccinations, urgences, saison, mois, moyennes mobiles, lags
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='padding: 16px; background: #F9FAFB; border-radius: 12px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='font-weight: 500;'>Fenêtre temporelle</span>
            <span style='color: #2563eb;'>3 semaines</span>
        </div>
        <div style='font-size: 13px; color: #6B7280;'>
            Analyse des 3 dernières semaines pour prédiction
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='padding: 16px; background: #F9FAFB; border-radius: 12px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='font-weight: 500;'>Mise à jour</span>
            <span style='color: #2563eb;'>Hebdomadaire</span>
        </div>
        <div style='font-size: 13px; color: #6B7280;'>
            Réentraînement automatique avec nouvelles données
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='padding: 16px; background: #F9FAFB; border-radius: 12px; margin-bottom: 12px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='font-weight: 500;'>Erreur moyenne (MAE)</span>
            <span style='color: #2563eb;'>±32 passages</span>
        </div>
        <div style='font-size: 13px; color: #6B7280;'>
            Écart absolu moyen entre prévision et réalité
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ===== RÉSULTATS DE VALIDATION =====
st.markdown("### 🎯 Résultats de validation")

val_col1, val_col2, val_col3, val_col4 = st.columns(4)

with val_col1:
    st.markdown("""
    <div style='text-align: center; padding: 24px; background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%); border-radius: 16px;'>
        <div style='font-size: 36px; font-weight: bold; color: #2563eb; margin-bottom: 8px;'>89%</div>
        <div style='font-size: 13px; color: #6B7280;'>Coefficient R²</div>
    </div>
    """, unsafe_allow_html=True)

with val_col2:
    st.markdown("""
    <div style='text-align: center; padding: 24px; background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border-radius: 16px;'>
        <div style='font-size: 36px; font-weight: bold; color: #10b981; margin-bottom: 8px;'>85%</div>
        <div style='font-size: 13px; color: #6B7280;'>Précision (Accuracy)</div>
    </div>
    """, unsafe_allow_html=True)

with val_col3:
    st.markdown("""
    <div style='text-align: center; padding: 24px; background: linear-gradient(135deg, #FDF4FF 0%, #FAE8FF 100%); border-radius: 16px;'>
        <div style='font-size: 36px; font-weight: bold; color: #a855f7; margin-bottom: 8px;'>32</div>
        <div style='font-size: 13px; color: #6B7280;'>MAE (passages)</div>
    </div>
    """, unsafe_allow_html=True)

with val_col4:
    st.markdown("""
    <div style='text-align: center; padding: 24px; background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%); border-radius: 16px;'>
        <div style='font-size: 36px; font-weight: bold; color: #f97316; margin-bottom: 8px;'>45</div>
        <div style='font-size: 13px; color: #6B7280;'>RMSE (passages)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Explication validation
st.markdown("""
<div style='padding: 20px; background: #F9FAFB; border-radius: 12px;'>
    <strong>📋 Méthode de validation :</strong> Validation croisée temporelle sur 3 saisons 
    épidémiques (2021-2024). Le modèle a été testé sur des données jamais vues pendant 
    l'entraînement pour garantir sa capacité de généralisation.
    
    <br><br>
    
    <strong>🎓 Interprétation des métriques :</strong>
    <ul>
        <li><strong>R² = 0.89</strong> : Le modèle explique 89% de la variance des données (excellent)</li>
        <li><strong>Accuracy = 85%</strong> : 85% des prédictions de niveau d'alerte sont correctes</li>
        <li><strong>MAE = 32</strong> : En moyenne, le modèle se trompe de ±32 passages</li>
        <li><strong>RMSE = 45</strong> : Écart-type des erreurs (pénalise plus les grosses erreurs)</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ===== FEATURE IMPORTANCE =====
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔑 Importance des variables")

# Données d'importance
feature_importance = pd.DataFrame({
    'feature': ['Vaccinations S-1', 'Vaccinations S-2', 'Urgences S-1', 'Mois', 'Moyenne mobile 7j', 'Saison hiver'],
    'importance': [0.35, 0.25, 0.18, 0.10, 0.08, 0.04]
})

# Bar chart horizontal
fig_importance = go.Figure()

fig_importance.add_trace(go.Bar(
    y=feature_importance['feature'],
    x=feature_importance['importance'],
    orientation='h',
    marker=dict(
        color=feature_importance['importance'],
        colorscale='Blues',
        showscale=False
    ),
    text=[f"{val:.0%}" for val in feature_importance['importance']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Importance: %{x:.1%}<extra></extra>'
))

fig_importance.update_layout(
    xaxis_title="Importance relative",
    yaxis_title="",
    height=350,
    template="plotly_white",
    xaxis=dict(tickformat='.0%')
)

st.plotly_chart(fig_importance, use_container_width=True)

st.info("""
**💡 Interprétation :** Les vaccinations de la semaine précédente (S-1) sont le prédicteur le plus important, 
représentant 35% de l'importance du modèle. Cela confirme notre hypothèse : 
**plus on vaccine, moins il y a de passages aux urgences 2-3 semaines plus tard**.
""")


# ===== FOOTER =====
st.markdown("---")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.caption("Modèle : Random Forest | Entraînement : 2021-2024 | Validation : Cross-validation temporelle")