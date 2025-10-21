#!/usr/bin/env python3
"""
Script de merge des 3 fichiers de données épidémiologiques
Crée un dataframe unifié au niveau RÉGION × SEMAINE pour le ML et la visualisation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MAPPINGS GÉOGRAPHIQUES
# ============================================================================

# Mapping département → région (nouvelles régions depuis 2016)
DEPT_TO_REGION = {
    '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes',
    '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
    '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
    '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',

    '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
    '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
    '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',

    '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',

    '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
    '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',

    '2A': 'Corse', '2B': 'Corse',

    '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
    '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
    '68': 'Grand Est', '88': 'Grand Est',

    '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
    '62': 'Hauts-de-France', '80': 'Hauts-de-France',

    '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France',
    '91': 'Île-de-France', '92': 'Île-de-France', '93': 'Île-de-France',
    '94': 'Île-de-France', '95': 'Île-de-France',

    '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',

    '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
    '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
    '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
    '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',

    '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
    '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
    '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',

    '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
    '72': 'Pays de la Loire', '85': 'Pays de la Loire',

    '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
    '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
    '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',

    # DOM-TOM
    '971': 'Guadeloupe', '972': 'Martinique', '973': 'Guyane',
    '974': 'Réunion', '976': 'Mayotte'
}

# Mapping anciennes régions (IAS) → nouvelles régions (standardisées)
OLD_TO_NEW_REGION = {
    'Alsace': 'Grand Est',
    'Aquitaine': 'Nouvelle-Aquitaine',
    'Auvergne': 'Auvergne-Rhône-Alpes',
    'Basse-Normandie': 'Normandie',
    'Bourgogne': 'Bourgogne-Franche-Comté',
    'Bretagne': 'Bretagne',
    'Centre': 'Centre-Val de Loire',
    'Champagne-Ardenne': 'Grand Est',
    'Corse': 'Corse',
    'Franche-Comté': 'Bourgogne-Franche-Comté',
    'Haute-Normandie': 'Normandie',
    'Île-de-France': 'Île-de-France',
    'Languedoc-Roussillon': 'Occitanie',
    'Limousin': 'Nouvelle-Aquitaine',
    'Lorraine': 'Grand Est',
    'Midi-Pyrénées': 'Occitanie',
    'Nord-Pas-de-Calais': 'Hauts-de-France',
    'Pays de la Loire': 'Pays de la Loire',
    'Picardie': 'Hauts-de-France',
    'Poitou-Charentes': 'Nouvelle-Aquitaine',
    'Provence-Alpes-Côte d\'Azur': 'Provence-Alpes-Côte d\'Azur',
    'Rhône-Alpes': 'Auvergne-Rhône-Alpes',
}

# Standardisation des noms de régions dans urgences_clean
URGENCES_REGION_MAPPING = {
    'Auvergne et Rhône-Alpes': 'Auvergne-Rhône-Alpes',
    'Bourgogne et Franche-Comté': 'Bourgogne-Franche-Comté',
    'Nouvelle Aquitaine': 'Nouvelle-Aquitaine',
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_week_start(date):
    """Retourne le lundi de la semaine pour une date donnée"""
    if pd.isna(date):
        return pd.NaT
    return date - timedelta(days=date.weekday())


def add_temporal_features(df, date_col='date_semaine'):
    """Ajoute des features temporelles au dataframe"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df['annee'] = df[date_col].dt.year
    df['mois'] = df[date_col].dt.month
    df['semaine_annee'] = df[date_col].dt.isocalendar().week
    df['jour_annee'] = df[date_col].dt.dayofyear

    # Saison épidémiologique (octobre-mars = hiver, avril-sept = été)
    df['saison'] = df['mois'].apply(lambda m: 'hiver' if m in [10, 11, 12, 1, 2, 3] else 'ete')

    # Année épidémiologique (commence en octobre)
    df['annee_epidemio'] = df.apply(
        lambda row: row['annee'] if row['mois'] < 10 else row['annee'] + 1,
        axis=1
    )

    return df


# ============================================================================
# CHARGEMENT ET PREPROCESSING
# ============================================================================

def load_and_process_grippe():
    """Charge et traite les données de vaccination grippe"""
    print("📊 Chargement des données de vaccination grippe...")

    df = pd.read_csv('data/processed/grippe_departement.csv')

    # Renommer les colonnes
    df.columns = ['annee', 'dept_code', 'dept_name', 'vacc_moins_65_risque',
                  'vacc_65_plus', 'vacc_65_74', 'vacc_75_plus']

    # Mapper vers les nouvelles régions
    df['region'] = df['dept_code'].astype(str).map(DEPT_TO_REGION)

    # Supprimer les départements sans mapping (DOM-TOM non présents dans urgences)
    df = df.dropna(subset=['region'])

    # Agréger par région et année (moyenne des taux de vaccination)
    df_agg = df.groupby(['annee', 'region']).agg({
        'vacc_moins_65_risque': 'mean',
        'vacc_65_plus': 'mean',
        'vacc_65_74': 'mean',
        'vacc_75_plus': 'mean'
    }).reset_index()

    print(f"   ✓ {len(df_agg)} lignes (région × année)")
    return df_agg


def load_and_process_ias():
    """
    Charge et traite les données IAS (Indicateur Activité Syndromique)
    Stratégie hybride : IAS national (2018-2024) + IAS régional (2023-2024)
    """
    print("📊 Chargement des données IAS (stratégie hybride)...")

    # ========== 1. IAS NATIONAL (2018-2024) ==========
    print("   • IAS national (2018-2024)...")
    df_national = pd.read_csv('data/processed/ias_national.csv')
    df_national['date'] = pd.to_datetime(df_national['date'])
    df_national['date_semaine'] = df_national['date'].apply(get_week_start)

    # Agréger par semaine (moyenne hebdomadaire)
    df_national_week = df_national.groupby('date_semaine').agg({
        'ias_national': 'mean'
    }).reset_index()

    print(f"     ✓ {len(df_national_week)} semaines")

    # ========== 2. IAS RÉGIONAL (2023-2024) ==========
    print("   • IAS régional (2023-2024)...")
    df_regional = pd.read_csv('data/processed/ias_regional.csv')
    df_regional['date'] = pd.to_datetime(df_regional['date'])
    df_regional['date_semaine'] = df_regional['date'].apply(get_week_start)

    # Agréger par semaine et région
    df_regional_week = df_regional.groupby(['date_semaine', 'region']).agg({
        'ias_regional': ['mean', 'std', 'min', 'max']
    }).reset_index()

    # Flatten columns
    df_regional_week.columns = ['date_semaine', 'region', 'taux_ias_moyen', 'taux_ias_std',
                                 'taux_ias_min', 'taux_ias_max']

    print(f"     ✓ {len(df_regional_week)} lignes (région × semaine)")
    print(f"     ✓ {df_regional_week['region'].nunique()} régions")

    # ========== 3. COMBINER : National comme base, Régional en override ==========
    # On retourne les deux dataframes pour le merge
    return df_national_week, df_regional_week


def load_and_process_urgences():
    """Charge et traite les données urgences et SOS médecins"""
    print("📊 Chargement des données urgences...")

    df = pd.read_csv('data/processed/urgences_clean.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Standardiser les noms de régions
    df['region'] = df['region'].replace(URGENCES_REGION_MAPPING)

    # Les données sont déjà hebdomadaires, on calcule juste le lundi de la semaine
    df['date_semaine'] = df['date'].apply(get_week_start)

    # Renommer pour clarté
    df = df.rename(columns={
        'urgences': 'urgences_grippe',
        'sos': 'sos_medecins'
    })

    df = df[['date_semaine', 'region', 'urgences_grippe', 'sos_medecins']]

    print(f"   ✓ {len(df)} lignes (région × semaine)")
    return df


# ============================================================================
# MERGE PRINCIPAL
# ============================================================================

def merge_all_data():
    """Merge les 3 sources de données en un dataframe unifié avec stratégie hybride IAS"""
    print("\n🔗 MERGE DES DONNÉES\n" + "="*60)

    # Chargement
    df_grippe = load_and_process_grippe()
    df_ias_national, df_ias_regional = load_and_process_ias()
    df_urgences = load_and_process_urgences()

    print("\n🔀 Fusion des datasets...")

    # ========== STRATÉGIE HYBRIDE IAS ==========
    # 1. Merge urgences avec IAS national (toutes régions reçoivent la même valeur)
    print("   • Étape 1/3 : Urgences + IAS national...")
    df = pd.merge(
        df_urgences,
        df_ias_national,
        on='date_semaine',
        how='left'
    )
    print(f"     ✓ {len(df)} lignes")

    # 2. Merge avec IAS régional (override quand disponible)
    print("   • Étape 2/3 : Ajout IAS régional (override)...")
    df = pd.merge(
        df,
        df_ias_regional,
        on=['date_semaine', 'region'],
        how='left'
    )

    # 3. Créer colonne IAS hybride : régional si dispo, sinon national
    df['taux_ias'] = df['taux_ias_moyen'].fillna(df['ias_national'])

    # Garder les stats régionales quand disponibles
    df['taux_ias_std'] = df['taux_ias_std'].fillna(0)
    df['taux_ias_min'] = df['taux_ias_min'].fillna(df['taux_ias'])
    df['taux_ias_max'] = df['taux_ias_max'].fillna(df['taux_ias'])

    # Supprimer colonnes temporaires
    df = df.drop(columns=['ias_national', 'taux_ias_moyen'])

    print(f"     ✓ IAS hybride créé (régional prioritaire)")

    # Ajouter l'année pour le merge avec grippe
    df['annee'] = pd.to_datetime(df['date_semaine']).dt.year

    # 4. Merge avec vaccination grippe
    print("   • Étape 3/3 : Ajout vaccination...")
    df = pd.merge(
        df,
        df_grippe,
        on=['annee', 'region'],
        how='left'
    )

    print(f"   ✓ Total final : {len(df)} lignes")

    # Réordonner les colonnes
    cols_order = [
        'date_semaine', 'region', 'annee',
        'taux_ias', 'taux_ias_std', 'taux_ias_min', 'taux_ias_max',
        'urgences_grippe', 'sos_medecins',
        'vacc_moins_65_risque', 'vacc_65_plus', 'vacc_65_74', 'vacc_75_plus'
    ]
    df = df[cols_order]

    # Trier par date et région
    df = df.sort_values(['date_semaine', 'region']).reset_index(drop=True)

    return df


def add_features_and_clean(df):
    """Ajoute des features et nettoie le dataframe final"""
    print("\n🎨 FEATURE ENGINEERING\n" + "="*60)

    # Features temporelles
    df = add_temporal_features(df, 'date_semaine')

    # Forward fill pour la vaccination (annuelle → hebdo)
    vacc_cols = ['vacc_moins_65_risque', 'vacc_65_plus', 'vacc_65_74', 'vacc_75_plus']
    df[vacc_cols] = df.groupby('region')[vacc_cols].ffill()

    print(f"   ✓ Features temporelles ajoutées (saison, semaine_annee, etc.)")

    # Statistiques de qualité
    print(f"\n📈 STATISTIQUES FINALES\n" + "="*60)
    print(f"   • Période : {df['date_semaine'].min()} → {df['date_semaine'].max()}")
    print(f"   • Régions : {df['region'].nunique()} régions")
    print(f"   • Total lignes : {len(df)}")
    print(f"\n   • Valeurs manquantes :")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, count in missing.items():
        pct = 100 * count / len(df)
        print(f"     - {col}: {count} ({pct:.1f}%)")

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  MERGE DONNÉES ÉPIDÉMIOLOGIQUES - HACKATHON")
    print("="*60 + "\n")

    # Merge
    df = merge_all_data()

    # Features
    df = add_features_and_clean(df)

    # Sauvegarde
    output_path = 'data/processed/master_dataframe.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✅ Fichier sauvegardé : {output_path}")
    print(f"   📦 Shape : {df.shape}")
    print(f"\n{'='*60}\n")

    # Aperçu
    print("📋 APERÇU DES DONNÉES (5 premières lignes) :\n")
    print(df.head().to_string())

    return df


if __name__ == "__main__":
    df = main()
