# Master DataFrame - Documentation

## Fichier : `master_dataframe.csv`

Dataframe unifié au niveau **RÉGION × SEMAINE** pour prédire les vagues épidémiques de grippe.

---

## 📊 Structure du fichier

### Dimensions
- **5258 lignes** (région × semaine)
- **18 colonnes**
- **Période** : 2019-12-30 → 2025-10-06
- **18 régions** françaises

### Colonnes

| Colonne | Type | Description | Source |
|---------|------|-------------|--------|
| `date_semaine` | date | Date du lundi de la semaine | Calculé |
| `region` | string | Nom de la région (nouvelles régions 2016) | Standardisé |
| `annee` | int | Année | Calculé |
| `taux_ias` | float | **IAS HYBRIDE** : Régional si dispo, sinon National | ias_national.csv + ias_regional.csv |
| `taux_ias_std` | float | Écart-type IAS (régional uniquement) | ias_regional.csv |
| `taux_ias_min` | float | Taux IAS min (régional uniquement) | ias_regional.csv |
| `taux_ias_max` | float | Taux IAS max (régional uniquement) | ias_regional.csv |
| `urgences_grippe` | float | Passages urgences pour grippe | urgences_clean.csv |
| `sos_medecins` | float | Appels SOS médecins | urgences_clean.csv |
| `vacc_moins_65_risque` | float | Taux vaccination <65 ans à risque (%) | grippe_departement.csv |
| `vacc_65_plus` | float | Taux vaccination 65+ (%) | grippe_departement.csv |
| `vacc_65_74` | float | Taux vaccination 65-74 ans (%) | grippe_departement.csv |
| `vacc_75_plus` | float | Taux vaccination 75+ (%) | grippe_departement.csv |
| `mois` | int | Mois (1-12) | Calculé |
| `semaine_annee` | int | Semaine ISO (1-53) | Calculé |
| `jour_annee` | int | Jour de l'année (1-366) | Calculé |
| `saison` | string | Saison épidémio ('hiver' ou 'ete') | Calculé |
| `annee_epidemio` | int | Année épidémiologique (oct→oct) | Calculé |

---

## 🔍 Valeurs manquantes

| Variable | Manquantes | % |
|----------|------------|---|
| `vacc_65_74` | 2179 | 41.4% |
| `vacc_75_plus` | 2179 | 41.4% |
| `taux_ias` | 1134 | 21.6% |
| `taux_ias_min/max` | 1134 | 21.6% |
| `vacc_moins_65_risque` | 604 | 11.5% |
| `vacc_65_plus` | 604 | 11.5% |

**Explications** :
- **IAS** : ✅ **Stratégie hybride** - National (2018-2024) + Régional quand disponible (2023-2024)
  - **78.4% de couverture** (4124/5258 lignes)
  - Période : déc 2019 → juil 2024
  - Avant déc 2019 : pas de données urgences donc pas d'IAS
- **Vaccination** : Données annuelles (forward fill appliqué)
- **Urgences/SOS** : Complet sur toute la période

---

## 🎯 Utilisation pour votre app

### 1. Chargement
```python
import pandas as pd

df = pd.read_csv('data/processed/master_dataframe.csv')
df['date_semaine'] = pd.to_datetime(df['date_semaine'])
```

### 2. Filtrage par région
```python
# Une région
df_idf = df[df['region'] == 'Île-de-France']

# Plusieurs régions
regions_interet = ['Île-de-France', 'Auvergne-Rhône-Alpes', 'Provence-Alpes-Côte d\'Azur']
df_filtered = df[df['region'].isin(regions_interet)]
```

### 3. Filtrage temporel
```python
# Dernière année
df_recent = df[df['date_semaine'] >= '2024-01-01']

# Hiver uniquement
df_hiver = df[df['saison'] == 'hiver']

# Période spécifique
df_periode = df[(df['date_semaine'] >= '2023-10-01') &
                 (df['date_semaine'] < '2024-04-01')]
```

### 4. Agrégation nationale
```python
# Moyenne France par semaine
df_national = df.groupby('date_semaine').agg({
    'urgences_grippe': 'mean',
    'sos_medecins': 'mean',
    'taux_ias_moyen': 'mean'
}).reset_index()
```

### 5. Détection de pics épidémiques
```python
# Seuil d'alerte : 95e percentile
seuil_urgences = df['urgences_grippe'].quantile(0.95)
df_pics = df[df['urgences_grippe'] > seuil_urgences]

print(f"Nombre de semaines en alerte : {len(df_pics)}")
```

### 6. Visualisation Streamlit
```python
import streamlit as st
import plotly.express as px

# Sélecteur de région
region = st.selectbox('Région', df['region'].unique())

# Filtrer
df_region = df[df['region'] == region]

# Graphique
fig = px.line(df_region, x='date_semaine', y='urgences_grippe',
              title=f'Urgences grippe - {region}')
st.plotly_chart(fig)
```

---

## 🤖 Utilisation pour le Machine Learning

### Variables target (à prédire)
- `urgences_grippe` : Nombre de passages urgences
- `sos_medecins` : Nombre d'appels SOS

### Features potentielles
- **Temporelles** : `mois`, `semaine_annee`, `saison`, `annee_epidemio`
- **IAS** : `taux_ias_moyen`, `taux_ias_std`
- **Vaccination** : `vacc_65_plus`, `vacc_moins_65_risque`
- **Géographiques** : `region` (encoding)
- **Lags** : Créer features `urgences_grippe_lag_1`, `urgences_grippe_lag_2`, etc.

### Exemple : Feature engineering
```python
# Lags (valeurs des semaines précédentes)
for i in [1, 2, 3, 4]:
    df[f'urgences_lag_{i}'] = df.groupby('region')['urgences_grippe'].shift(i)

# Rolling mean (moyenne mobile)
df['urgences_rolling_4w'] = df.groupby('region')['urgences_grippe'].rolling(4).mean().reset_index(0, drop=True)

# Différence semaine précédente
df['urgences_diff'] = df.groupby('region')['urgences_grippe'].diff()
```

### Préparation dataset ML
```python
# Supprimer NaN créés par lags
df_ml = df.dropna(subset=['urgences_grippe', 'urgences_lag_1'])

# Features
X = df_ml[['mois', 'semaine_annee', 'urgences_lag_1', 'urgences_lag_2',
           'vacc_65_plus', 'taux_ias_moyen']]

# Target
y = df_ml['urgences_grippe']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
```

---

## 📈 Exemples de graphiques pour l'app

### 1. Carte de France (heatmap régions)
```python
import plotly.express as px

# Moyenne dernière année par région
df_recent = df[df['date_semaine'] >= '2024-01-01']
avg_by_region = df_recent.groupby('region')['urgences_grippe'].mean().reset_index()

fig = px.choropleth(avg_by_region,
                    locations='region',
                    locationmode='country names',
                    color='urgences_grippe',
                    title='Urgences grippe moyenne par région (2024)')
```

### 2. Évolution temporelle multi-régions
```python
top5 = df.groupby('region')['urgences_grippe'].mean().nlargest(5).index
df_top5 = df[df['region'].isin(top5)]

fig = px.line(df_top5, x='date_semaine', y='urgences_grippe',
              color='region', title='Top 5 régions - Urgences grippe')
```

### 3. Box plot saisonnalité
```python
fig = px.box(df, x='mois', y='urgences_grippe',
             title='Distribution urgences par mois')
```

### 4. Prédictions vs réel
```python
# Après entraînement modèle
df_test['predictions'] = model.predict(X_test)

fig = px.line(df_test, x='date_semaine', y=['urgences_grippe', 'predictions'],
              title='Prédictions vs Réel')
```

---

## 🔄 Regenerer le fichier

### Étape 1 : Processing IAS (optionnel si déjà fait)
```bash
docker-compose exec streamlit python scripts/process_ias.py
```

Génère :
- `data/processed/ias_national.csv` (IAS France entière, 2018-2024)
- `data/processed/ias_regional.csv` (IAS par région, 2023-2024)

### Étape 2 : Merge final
```bash
docker-compose exec streamlit python scripts/merge_data.py
```

Le script :
1. Charge les 3 fichiers sources (grippe, ias, urgences)
2. **Applique stratégie hybride IAS** :
   - IAS national → répliqué sur toutes régions (2018-2024)
   - IAS régional → override quand disponible (2023-2024)
3. Standardise les noms de régions
4. Agrège département → région
5. Agrège jour → semaine
6. Merge sur (date_semaine, region)
7. Ajoute features temporelles
8. Exporte `master_dataframe.csv`

---

## ✅ Avantages de ce format

- ✅ **Granularité hebdomadaire** : Idéale pour prédire vagues épidémiques
- ✅ **18 régions** : Couverture complète France métropolitaine
- ✅ **6 ans de données** : Suffisant pour entraîner modèle
- ✅ **Features multiples** : IAS, urgences, SOS, vaccination
- ✅ **Features temporelles** : Saisonnalité automatique
- ✅ **Prêt pour Streamlit** : Facile à filtrer et visualiser
- ✅ **Prêt pour ML** : Structure tabular standard

---

## 📝 Notes

- **Stratégie hybride IAS** : National pour toutes périodes/régions, régional en override (2023-2024)
- Les taux de vaccination sont annuels → Forward fill pour hebdo
- Les urgences/SOS sont la variable la plus fiable (2019-2025)
- Pour prédictions : Focus sur `urgences_grippe` comme target
- Les DOM-TOM (Guadeloupe, Martinique, etc.) sont inclus

---

**Généré le** : 2025-10-21
**Scripts** :
- `scripts/process_ias.py` (processing IAS)
- `scripts/merge_data.py` (merge final)

**Sources** :
- `data/processed/grippe_departement.csv`
- `data/processed/ias_national.csv` (⭐ IAS 2018-2024)
- `data/processed/ias_regional.csv` (IAS régional 2023-2024)
- `data/processed/urgences_clean.csv`
