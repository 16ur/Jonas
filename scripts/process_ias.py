#!/usr/bin/env python3
"""
Script de processing des données IAS (Indicateur Activité Syndromique)
Traite les données nationales et régionales pour créer un fichier unifié
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*70)
print("  PROCESSING IAS - DONNÉES NATIONALES ET RÉGIONALES")
print("="*70 + "\n")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# ============================================================================
# MAPPING ANCIENNES RÉGIONS → NOUVELLES RÉGIONS
# ============================================================================

# Codes régions dans Openhealth (anciennes régions, codes INSEE)
REGION_CODE_TO_NAME = {
    '11': 'Île-de-France',
    '21': 'Champagne-Ardenne',
    '22': 'Picardie',
    '23': 'Haute-Normandie',
    '24': 'Centre',
    '25': 'Basse-Normandie',
    '26': 'Bourgogne',
    '31': 'Nord-Pas-de-Calais',
    '41': 'Lorraine',
    '42': 'Alsace',
    '43': 'Franche-Comté',
    '52': 'Pays de la Loire',
    '53': 'Bretagne',
    '54': 'Poitou-Charentes',
    '72': 'Aquitaine',
    '73': 'Midi-Pyrénées',
    '74': 'Limousin',
    '82': 'Rhône-Alpes',
    '83': 'Auvergne',
    '91': 'Languedoc-Roussillon',
    '93': 'Provence-Alpes-Côte d\'Azur',
    '94': 'Corse',
}

# Mapping anciennes → nouvelles régions
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


# ============================================================================
# PROCESSING IAS NATIONAL
# ============================================================================

def process_ias_national():
    """Traite les données IAS nationales (2018-2024)"""
    print("📊 1/2 - PROCESSING IAS NATIONAL")
    print("-" * 70)

    filepath = RAW_DIR / 'Openhealth_S-Grippal.csv'

    if not filepath.exists():
        print(f"   ❌ Fichier non trouvé : {filepath}")
        return None

    # Charger avec encoding latin-1 (caractères accentués)
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')

    print(f"   • Lignes chargées : {len(df)}")
    print(f"   • Colonnes : {df.columns.tolist()}")
    print(f"   • Période : {df['PERIODE'].min()} → {df['PERIODE'].max()}")

    # Nettoyer
    df['date'] = pd.to_datetime(df['PERIODE'])

    # Convertir IAS_lissé (format français avec virgule)
    df['ias_national'] = df['IAS_lissé'].astype(str).str.replace(',', '.').astype(float)

    # Garder uniquement date et IAS
    df_clean = df[['date', 'ias_national']].copy()

    # Supprimer les valeurs manquantes
    df_clean = df_clean.dropna(subset=['ias_national'])

    print(f"   • Lignes après nettoyage : {len(df_clean)}")
    print(f"   • Stats IAS national :")
    print(f"     - Min  : {df_clean['ias_national'].min():.2f}")
    print(f"     - Max  : {df_clean['ias_national'].max():.2f}")
    print(f"     - Mean : {df_clean['ias_national'].mean():.2f}")

    # Sauvegarder
    output_path = PROCESSED_DIR / 'ias_national.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"   ✅ Sauvegardé : {output_path}\n")

    return df_clean


# ============================================================================
# PROCESSING IAS RÉGIONAL
# ============================================================================

def process_ias_regional():
    """Traite les données IAS régionales (nov 2023 - juil 2024)"""
    print("📊 2/2 - PROCESSING IAS RÉGIONAL")
    print("-" * 70)

    filepath = RAW_DIR / 'Openhealth_S-Grippal_Regions.csv'

    if not filepath.exists():
        print(f"   ❌ Fichier non trouvé : {filepath}")
        return None

    # Charger
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')

    print(f"   • Lignes chargées : {len(df)}")
    print(f"   • Période : {df['PERIODE'].min()} → {df['PERIODE'].max()}")

    # Supprimer les lignes vides
    df = df[df['PERIODE'].notna() & (df['PERIODE'] != '')]

    # Parser la date (format dd-mm-yyyy)
    df['date'] = pd.to_datetime(df['PERIODE'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date'])

    # Extraire les colonnes régionales (Loc_Reg*)
    region_cols = [col for col in df.columns if col.startswith('Loc_Reg')]

    print(f"   • Colonnes régionales trouvées : {len(region_cols)}")

    # Transformer en format long (une ligne par région par jour)
    dfs = []

    for col in region_cols:
        # Extraire le code région (ex: Loc_Reg42 → 42)
        region_code = col.replace('Loc_Reg', '')

        if region_code not in REGION_CODE_TO_NAME:
            continue

        old_region_name = REGION_CODE_TO_NAME[region_code]
        new_region_name = OLD_TO_NEW_REGION.get(old_region_name, old_region_name)

        # Créer un dataframe pour cette région
        df_region = df[['date', col]].copy()
        df_region.columns = ['date', 'ias_regional']
        df_region['region'] = new_region_name

        # Convertir IAS (format français)
        df_region['ias_regional'] = (
            df_region['ias_regional']
            .astype(str)
            .str.replace(',', '.')
            .replace('NA', np.nan)
        )
        df_region['ias_regional'] = pd.to_numeric(df_region['ias_regional'], errors='coerce')

        # Supprimer les NA
        df_region = df_region.dropna(subset=['ias_regional'])

        dfs.append(df_region)

    # Concaténer toutes les régions
    df_all = pd.concat(dfs, ignore_index=True)

    print(f"   • Lignes après transformation : {len(df_all)}")
    print(f"   • Régions uniques : {df_all['region'].nunique()}")
    print(f"   • Régions : {sorted(df_all['region'].unique())}")

    # Agréger les anciennes régions vers nouvelles (moyenne)
    df_grouped = df_all.groupby(['date', 'region']).agg({
        'ias_regional': 'mean'
    }).reset_index()

    print(f"   • Lignes après agrégation : {len(df_grouped)}")
    print(f"   • Stats IAS régional :")
    print(f"     - Min  : {df_grouped['ias_regional'].min():.2f}")
    print(f"     - Max  : {df_grouped['ias_regional'].max():.2f}")
    print(f"     - Mean : {df_grouped['ias_regional'].mean():.2f}")

    # Sauvegarder
    output_path = PROCESSED_DIR / 'ias_regional.csv'
    df_grouped.to_csv(output_path, index=False)
    print(f"   ✅ Sauvegardé : {output_path}\n")

    return df_grouped


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Processing
    df_national = process_ias_national()
    df_regional = process_ias_regional()

    # Résumé
    print("="*70)
    print("✅ PROCESSING TERMINÉ")
    print("="*70)

    if df_national is not None:
        print(f"\n📊 IAS National :")
        print(f"   • {len(df_national)} jours")
        print(f"   • {df_national['date'].min()} → {df_national['date'].max()}")

    if df_regional is not None:
        print(f"\n📊 IAS Régional :")
        print(f"   • {len(df_regional)} lignes (région × jour)")
        print(f"   • {df_regional['date'].min()} → {df_regional['date'].max()}")
        print(f"   • {df_regional['region'].nunique()} régions")

    print(f"\n📁 Fichiers générés dans : {PROCESSED_DIR}/")
    print("   • ias_national.csv")
    print("   • ias_regional.csv")

    print("\n🎯 Prochaine étape : docker-compose exec streamlit python scripts/merge_data.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
