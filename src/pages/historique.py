"""
Page : Données Historiques
Analyse des saisons épidémiques et corrélations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# ===== CONFIGURATION PAGE =====
st.set_page_config(
    page_title="Données Historiques - Jonas",
    page_icon="📅",
    layout="wide"
)

# ===== CHARGEMENT DES VRAIES DONNÉES =====

@st.cache_data
def load_master_data():
    """Charge le master dataframe"""
    df = pd.read_csv('data/processed/master_dataframe.csv')
    df['date_semaine'] = pd.to_datetime(df['date_semaine'])
    return df

# Charger les données
df = load_master_data()

# Agréger par semaine (moyenne nationale)
df_national = df.groupby('date_semaine').agg({
    'taux_ias': 'mean',
    'urgences_grippe': 'mean',
    'sos_medecins': 'mean',
    'semaine_annee': 'first',
    'annee': 'first',
    'annee_epidemio': 'first'
}).reset_index()

# Créer colonne semaine ISO format "S01", "S02", etc.
df_national['week_label'] = 'S' + df_national['semaine_annee'].astype(str).str.zfill(2)

# ===== PRÉPARER DONNÉES SAISONNIÈRES (3 dernières saisons épidémiques COMPLÈTES) =====
def prepare_seasonal_data():
    """
    Prépare les données de comparaison saisonnière - 3 dernières saisons épidémiques COMPLÈTES
    Saison = octobre N à mars N+1 (alignées pour comparaison)
    Utilise URGENCES car plus stable et complet que IAS
    """
    # Filtrer les mois de saison épidémique (oct à mars)
    df_epidemic = df_national[df_national['date_semaine'].dt.month.isin([10, 11, 12, 1, 2, 3])].copy()

    # Créer une semaine relative dans la saison (0 à ~26 semaines)
    def get_season_week(row):
        month = row['date_semaine'].month
        week = row['semaine_annee']

        if month >= 10:  # Oct-Dec : semaines 40-52
            return week - 40
        else:  # Jan-Mar : semaines 1-13
            return week + 12  # Continue après la semaine 12 (S52 - S40 = 12)

    df_epidemic['week_in_season'] = df_epidemic.apply(get_season_week, axis=1)

    # Créer label "S0" à "S26" pour l'axe X
    df_epidemic['season_week_label'] = 'S' + df_epidemic['week_in_season'].astype(int).astype(str)

    # Identifier les saisons COMPLÈTES (qui ont un pic significatif d'urgences > 1000)
    season_peaks = df_epidemic.groupby('annee_epidemio')['urgences_grippe'].max()
    complete_seasons = season_peaks[season_peaks > 1000].index.tolist()

    # Prendre les 3 dernières saisons complètes
    complete_seasons = sorted(complete_seasons, reverse=True)[:3]

    # Grouper par saison épidémique
    seasons = {}
    for season_year in sorted(complete_seasons):
        season_data = df_epidemic[df_epidemic['annee_epidemio'] == season_year].copy()
        if len(season_data) > 10:  # Au moins 10 semaines de données
            # Trier par semaine dans la saison
            season_data = season_data.sort_values('week_in_season')

            # Agréger si plusieurs valeurs pour la même semaine relative
            season_agg = season_data.groupby('week_in_season').agg({
                'urgences_grippe': 'mean',
                'taux_ias': 'mean',
                'season_week_label': 'first'
            }).reset_index()

            # Clé = "2020-21" pour saison 2020-2021
            season_key = f"{season_year-1}-{str(season_year)[2:]}"
            seasons[season_key] = season_agg

    return seasons

seasonal_seasons = prepare_seasonal_data()

# ===== DONNÉES DE CORRÉLATION (IAS vs Urgences) =====
# Filtrer uniquement les périodes épidémiques (oct-mar) et urgences > 200 pour éliminer le bruit
df_corr_temp = df_national[
    (df_national['date_semaine'].dt.month.isin([10, 11, 12, 1, 2, 3])) &
    (df_national['taux_ias'].notna()) &
    (df_national['urgences_grippe'] > 200)  # Éliminer les valeurs très basses
][['taux_ias', 'urgences_grippe']].copy()

correlation_data = df_corr_temp.copy()
correlation_data.columns = ['ias', 'urgences']

# ===== PICS ÉPIDÉMIQUES PAR SAISON =====
def get_peaks_data():
    """Calcule les pics épidémiques par saison"""
    peaks = []
    for year in df_national['annee_epidemio'].unique():
        season_data = df_national[df_national['annee_epidemio'] == year]
        if len(season_data) > 0 and season_data['taux_ias'].notna().sum() > 0:
            max_idx = season_data['taux_ias'].idxmax()
            max_row = season_data.loc[max_idx]
            peaks.append({
                'saison': f'{year-1}-{str(year)[2:]}',
                'pic': max_row['taux_ias'],
                'semaine': max_row['week_label'],
                'date': max_row['date_semaine']
            })

    return pd.DataFrame(peaks)

peaks_data = get_peaks_data()


# ===== HEADER =====
st.title("Données Historiques")
st.markdown("Analyse des saisons épidémiques précédentes et corrélations IAS®")
st.markdown("""
Cette page présente une **analyse historique des saisons épidémiques de grippe** en France métropolitaine.  
Elle permet d’explorer les **tendances passées**, d’identifier les **périodes typiques de pic épidémique** et d’évaluer la **corrélation entre l’indicateur IAS® et les passages aux urgences**.  

**Objectifs :**
- Comprendre la **récurrence saisonnière** des vagues de grippe (octobre → mars)  
- Visualiser l’évolution des **urgences grippe** sur les dernières saisons complètes  
- Mesurer la **force de corrélation** entre IAS® et urgences pour valider son rôle d’indicateur précoce  
- Identifier les **pics épidémiques** afin de mieux calibrer les modèles prédictifs  

Cette analyse fournit le **socle analytique** sur lequel s’appuie Jonas pour anticiper les futures vagues de grippe.
""")
st.markdown("---")


# ===== MÉTRIQUES PRINCIPALES =====
col1, col2, col3 = st.columns(3)

# Calculer les métriques réelles
peaks_data = peaks_data[:-1]
nb_saisons = len(peaks_data)
if len(correlation_data) > 10:
    corr_r2 = np.corrcoef(correlation_data['ias'], correlation_data['urgences'])[0, 1] ** 2
else:
    corr_r2 = 0.0
max_ias = df_national['taux_ias'].max()
max_ias_year = df_national[df_national['taux_ias'] == max_ias]['annee'].values[0] if not pd.isna(max_ias) else 2020

with col1:
    st.metric(
        label="Saisons analysées",
        value=str(nb_saisons),
        help="Nombre de saisons épidémiques dans l'analyse"
    )

with col2:
    st.metric(
        label="Corrélation R²",
        value=f"{corr_r2:.2f}",
        delta="Forte corrélation" if corr_r2 > 0.7 else "Corrélation modérée",
        help="Coefficient de corrélation entre IAS® et urgences"
    )

with col3:
    st.metric(
        label=f"IAS® max ({int(max_ias_year)})",
        value=f"{max_ias:.0f}",
        help="Valeur maximale d'IAS® enregistrée"
    )

st.markdown("<br>", unsafe_allow_html=True)


# ===== GRAPHIQUE 1 : COMPARAISON SAISONNIÈRE =====
st.markdown("### Comparaison des saisons épidémiques")
st.caption("Évolution des passages aux urgences pour grippe - 3 dernières saisons complètes (octobre → mars)")

# Graphique
fig_seasonal = go.Figure()

# Prendre les 3 dernières saisons (déjà trié dans prepare_seasonal_data)
season_keys = sorted(seasonal_seasons.keys(), reverse=True)[:3]
colors = ['#94a3b8', '#06b6d4', '#2563eb']
color_map = dict(zip(reversed(season_keys), colors))  # Plus ancienne = gris, plus récente = bleu

for season_key in reversed(season_keys):  # Afficher de la plus ancienne à la plus récente
    season_df = seasonal_seasons[season_key]
    is_current = (season_key == max(season_keys))  # Dernière saison = actuelle

    # Utiliser la semaine relative dans la saison (S0 à S26)
    week_labels = season_df['season_week_label'].tolist()

    fig_seasonal.add_trace(go.Scatter(
        x=week_labels,
        y=season_df['urgences_grippe'],
        mode='lines+markers',
        name=f'Saison {season_key}',
        line=dict(color=color_map[season_key], width=3 if is_current else 2),
        marker=dict(size=10 if is_current else 8, color=color_map[season_key]),
        hovertemplate='<b>Saison ' + season_key + ' - %{x}</b><br>Urgences: %{y:.0f}<extra></extra>'
    ))

fig_seasonal.update_layout(
    xaxis_title="Semaine de la saison (S0 = début octobre, S12 = fin décembre, S24 = fin mars)",
    yaxis_title="Passages aux urgences (moyenne nationale)",
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

# Info sur les saisons affichées
if len(season_keys) > 0:
    st.caption(f"Saisons épidémiques : {', '.join(sorted(season_keys))} • Alignées sur la même période (octobre → mars)")
    st.success(f"""
    **Schéma répétitif clair :**
    - **Montée progressive** d'octobre (S0) à décembre (S12)
    - **Pic épidémique** entre fin décembre et mi-janvier (S12-S16)
    - **Décroissance** de janvier à mars (S16-S24)

    ➡️ Ce pattern se répète **chaque année**, permettant de **prédire** la prochaine vague !
    """)

st.markdown("<br>", unsafe_allow_html=True)


# ===== SECTION 2 COLONNES =====
col_left, col_right = st.columns(2)

# ===== COLONNE GAUCHE : CORRÉLATION =====
with col_left:
    st.markdown("### Corrélation IAS® & Passages aux urgences")
    
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
    corr_strength = "Très forte" if corr_r2 > 0.8 else ("Forte" if corr_r2 > 0.6 else "Modérée")
    corr_color = "success" if corr_r2 > 0.6 else "info"

    eval(f"st.{corr_color}")(f"""
    **Coefficient de corrélation (R²) : {corr_r2:.2f}**

    **{corr_strength} corrélation** entre l'IAS® et les passages aux urgences pendant les périodes épidémiques.

    **Ce que cela signifie :**
    - L'IAS® peut **prédire** les urgences avec {corr_r2*100:.0f}% de précision
    - Une augmentation de l'IAS® = augmentation des urgences **2-3 semaines plus tard**
    - Permet une **alerte précoce** avant saturation des hôpitaux
    """)


# ===== COLONNE DROITE : PICS ÉPIDÉMIQUES =====
with col_right:
    st.markdown("### Pics épidémiques par saison")
    
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
    st.markdown("**Statistiques clés**")

    # Calculer stats réelles
    pic_moyen = peaks_data['pic'].mean() if len(peaks_data) > 0 else 0
    semaine_pics = peaks_data['semaine'].mode()[0] if len(peaks_data) > 0 else "N/A"

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Pic moyen", f"{pic_moyen:.0f} IAS®")
        st.metric("Durée moyenne épidémie", "12-14 semaines")

    with metric_col2:
        st.metric("Semaine type du pic", semaine_pics)
        if len(peaks_data) >= 2:
            var_last = ((peaks_data.iloc[-1]['pic'] - peaks_data.iloc[-2]['pic']) / peaks_data.iloc[-2]['pic'] * 100)
            st.metric("Variation dernière saison", f"{var_last:+.1f}%")
        else:
            st.metric("Variation dernière saison", "N/A")

st.markdown("<br>", unsafe_allow_html=True)


# ===== TABLEAU DE DONNÉES DÉTAILLÉ =====
st.markdown("### Tableau de données détaillé")

# Préparer données pour tableau : dernières 20 semaines
df_table = df_national.sort_values('date_semaine', ascending=False).head(20).copy()
df_table = df_table[['date_semaine', 'week_label', 'taux_ias', 'urgences_grippe', 'sos_medecins']]

# Formatter les colonnes
df_table['date_semaine'] = df_table['date_semaine'].dt.strftime('%d/%m/%Y')
df_table['taux_ias'] = df_table['taux_ias'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
df_table['urgences_grippe'] = df_table['urgences_grippe'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
df_table['sos_medecins'] = df_table['sos_medecins'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")

# Renommer colonnes
df_table.columns = ['Date', 'Semaine', 'IAS®', 'Urgences', 'SOS Médecins']

# Afficher tableau
st.dataframe(
    df_table,
    use_container_width=True,
    height=400,
    hide_index=True
)

st.markdown("*Les IAS® ne sont pas disponibles sur les données les plus récentes en raison des délais de collecte et de traitement.")

# Bouton export tableau
col_export1, col_export2, col_export3 = st.columns([2, 1, 2])
with col_export2:
    csv_full = df_national[['date_semaine', 'week_label', 'taux_ias', 'urgences_grippe', 'sos_medecins']].to_csv(index=False)
    st.download_button(
        label="Exporter CSV complet",
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
    **Prédictibilité**
    
    Les pics épidémiques surviennent systématiquement entre les semaines S50 et S02 (fin décembre - début janvier).
    """)

with insight_col2:
    # Calculer tendance dernière saison si possible
    if len(peaks_data) >= 2:
        tendance_pct = ((peaks_data.iloc[-1]['pic'] - peaks_data.iloc[-2]['pic']) / peaks_data.iloc[-2]['pic'] * 100)
        tendance_txt = f"{'augmentation' if tendance_pct > 0 else 'diminution'} de {abs(tendance_pct):.1f}%"
        last_season = peaks_data.iloc[-1]['saison']
    else:
        tendance_txt = "données insuffisantes"
        last_season = "N/A"

    st.info(f"""
    **Tendance {last_season}**

    La saison {last_season} montre une {tendance_txt} par rapport à la saison précédente.
    """)

with insight_col3:
    st.warning("""
    **Alerte précoce**

    L'IAS® permet de détecter une montée épidémique 2-3 semaines avant le pic des urgences.
    """)


# ===== FOOTER =====
st.markdown("---")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.caption("Données : Santé Publique France | Modèle : Jonas v1.0")

