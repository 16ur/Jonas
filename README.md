# Jonas - Prédiction des Vagues Épidémiques de Grippe

> **Application de prédiction et surveillance des épidémies de grippe en France**
> Développée pour le hackathon T-HAK-700 - Epitech 2025

---

## Objectif du Projet

**Jonas** permet de **prédire les vagues épidémiques de grippe** en analysant des données historiques (2018-2025) pour :
- 📈 **Anticiper** les pics d'urgences hospitalières 2-3 semaines à l'avance
- 🏥 **Optimiser** la gestion des ressources hospitalières
- 🔔 **Alerter** les autorités sanitaires avant saturation des urgences

**Public cible** : Décideurs hospitaliers, épidémiologistes, grand public

---

## Notre Approche

### Principe
Nous analysons **3 sources de données** pour prédire les épidémies :
1. **IAS® (Indicateur Activité Syndromique)** : Surveillance en temps réel des symptômes grippaux
2. **Urgences hospitalières** : Passages aux urgences pour grippe
3. **Vaccination** : Taux de couverture vaccinale par région

### Insight clé
Les épidémies de grippe suivent un **schéma répétitif chaque année** :
- **Montée progressive** : octobre → décembre
- **Pic épidémique** : fin décembre → mi-janvier
- **Décroissance** : janvier → mars

En détectant tôt les signaux (IAS en hausse), on peut **prédire** le pic d'urgences !

---

## Nos Données

### Sources
- **Santé Publique France** (urgences, SOS médecins, vaccination)
- **OpenHealth** (IAS national et régional)

### Couverture
- **Période** : 2018-2025 (6 ans de données)
- **Géographie** : 18 régions françaises
- **Granularité** : Hebdomadaire

### Notre défi : Merger des données hétérogènes

| Fichier | Période | Granularité | Problème |
|---------|---------|-------------|----------|
| IAS régional | nov 2023 - juil 2024 | Jour × Région | Période trop courte (9 mois) |
| IAS national | mai 2018 - juil 2024 | Jour × France | 6 ans mais pas de détail régional |
| Urgences | déc 2019 - oct 2025 | Semaine × Région | Complet |
| Vaccination | 2016 - 2024 | Année × Département | Long historique |

**Notre solution : Stratégie hybride**
- IAS national → répliqué sur toutes les régions (baseline)
- IAS régional → remplace le national quand disponible (2023-2024)
- **Résultat** : 78% de couverture IAS (vs 10% avant)

---

## Démarrage Rapide

### Prérequis
- Docker & Docker Compose
- Git

### Installation

```bash
# 1. Cloner le projet
git clone https://github.com/EpitechMscProPromo2027/T-HAK-700-MAR_25.git
cd T-HAK-700-MAR_25

# 2. Lancer l'application
docker-compose up --build

# 3. Ouvrir dans le navigateur
# → http://localhost:8501
```

C'est tout !

---

## Pipeline de Données

### Étape 1 : Téléchargement (optionnel)
```bash
docker-compose exec streamlit python scripts/dl_data.py
```
→ Télécharge les données depuis data.gouv.fr et OpenHealth

### Étape 2 : Nettoyage IAS
```bash
docker-compose exec streamlit python scripts/process_ias.py
```
→ Génère :
- `data/processed/ias_national.csv` (IAS France, 2018-2024)
- `data/processed/ias_regional.csv` (IAS par région, 2023-2024)

### Étape 3 : Merge & Features
```bash
docker-compose exec streamlit python scripts/merge_data.py
```
→ Fusionne toutes les sources et génère :
- `data/processed/master_dataframe.csv` (fichier final pour ML)

**Contenu du master dataframe** :
- 5258 lignes (région × semaine)
- 18 colonnes : IAS, urgences, SOS, vaccination, features temporelles
- Prêt pour entraînement de modèle de prédiction

---

## Structure du Projet

```
Jonas/
├── src/
│   ├── app.py                    # Application Streamlit principale
│   └── pages/
│       ├── historique.py         # Analyse historique & comparaisons
│       └── prediction.py         # Prédictions futures
├── scripts/
│   ├── dl_data.py                # Téléchargement données
│   ├── process_ias.py            # Nettoyage IAS (stratégie hybride)
│   ├── merge_data.py             # Fusion finale + feature engineering
│   └── cleaning_urgences.py      # Nettoyage urgences
├── data/
│   ├── raw/                      # Données brutes téléchargées
│   └── processed/                # Données nettoyées et mergées
│       ├── master_dataframe.csv  # ⭐ Fichier final exploitable
│       ├── ias_national.csv
│       ├── ias_regional.csv
│       └── README.md             # Documentation des données
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 💡 Choix Techniques

### Pourquoi une stratégie hybride IAS ?
**Problème initial** : L'IAS régional ne couvre que 9 mois (nov 2023 - juil 2024)
**Solution** : Combiner IAS national (6 ans) + IAS régional (quand disponible)
**Résultat** : **78% de couverture** au lieu de 10% → meilleure qualité prédictive

### Pourquoi agréger par semaine ?
- Cohérent avec le cycle épidémique (pics hebdomadaires)
- Réduit le bruit des données quotidiennes
- Compatible avec toutes les sources (vaccination annuelle → vaccination hebdo via forward fill)

### Pourquoi focus sur les saisons épidémiques (oct-mars) ?
- Les épidémies de grippe surviennent **exclusivement en hiver**
- Comparer octobre-mars permet de voir les **patterns répétitifs**
- Évite le bruit de l'été (IAS proche de 0)

---

## Résultats Clés

### Page Historique
- **3 courbes saisonnières** montrant le pattern répétitif
- **Corrélation IAS vs Urgences** : R² = 0.48-0.70 selon filtres
- **Pics épidémiques** identifiés automatiquement par saison

### Insights
- **100% des pics** surviennent entre fin décembre et mi-janvier (S12-S16)
- **Corrélation forte** : L'IAS peut prédire les urgences avec 48-70% de précision
- **Délai d'alerte** : 2-3 semaines entre montée IAS et pic urgences

---

## 👥 Équipe

Projet développé dans le cadre du hackathon **T-HAK-700** - Epitech MSc Pro 2027

---

## 📄 License

MIT License - Projet éducatif

---

## 🆘 Support

**Problème avec Docker ?**
```bash
# Reconstruire l'image
docker-compose build --no-cache

# Vérifier les logs
docker-compose logs -f
```

**Données manquantes ?**
```bash
# Régénérer toutes les données
docker-compose exec streamlit python scripts/process_ias.py
docker-compose exec streamlit python scripts/merge_data.py
```

---