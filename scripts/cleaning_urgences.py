"""
Nettoyage des données URGENCES
Objectif : Passer de départemental à régional, filtrer "Tous âges"
"""

import pandas as pd
import os
from pathlib import Path

print("="*80)
print("🏥 NETTOYAGE DES URGENCES")
print("="*80 + "\n")

# ===== DÉFINIR LES CHEMINS =====
# Déterminer le répertoire racine du projet
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Chemins absolus basés sur la racine du projet
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

print(f"📁 Répertoire du projet : {PROJECT_ROOT}")
print(f"📁 Données brutes : {RAW_DATA_DIR}")
print(f"📁 Données traitées : {PROCESSED_DATA_DIR}\n")

# ===== CHARGER =====
print("📂 Chargement...")
input_file = RAW_DATA_DIR / 'urgences_sos_departement.csv'
print(f"Lecture de : {input_file}")
df = pd.read_csv(input_file)
print(f"Lignes initiales : {len(df)}")
print(f"Colonnes : {df.columns.tolist()}\n")

# ===== AFFICHER CLASSES D'ÂGE DISPONIBLES =====
print("📋 Classes d'âge disponibles :")
print(df['Classe d\'âge'].unique())
print()

# ===== FILTRER "Tous âges" UNIQUEMENT =====
print("🔍 Filtrage 'Tous âges'...")
df_filtered = df[df['Classe d\'âge'] == 'Tous âges'].copy()
print(f"Lignes après filtrage : {len(df_filtered)}")

# Vérifier qu'on a bien des données
if len(df_filtered) == 0:
    print("⚠️ ATTENTION : Aucune ligne avec 'Tous âges'")
    print("Classes disponibles :", df['Classe d\'âge'].unique())
    print("\n💡 On va prendre TOUTES les classes et agréger")
    df_filtered = df.copy()

# ===== RENOMMER COLONNES =====
print("\n📝 Renommage des colonnes...")
df_clean = df_filtered.rename(columns={
    '1er jour de la semaine': 'date',
    'Région': 'region',
    'Taux de passages aux urgences pour grippe': 'urgences',
    'Taux d\'actes médicaux SOS médecins pour grippe': 'sos',
    'Département': 'departement'
})

# ===== CONVERTIR TYPES =====
print("🔧 Conversion des types...")
df_clean['date'] = pd.to_datetime(df_clean['date'])
df_clean['urgences'] = pd.to_numeric(df_clean['urgences'], errors='coerce')
df_clean['sos'] = pd.to_numeric(df_clean['sos'], errors='coerce')

# ===== AGRÉGER PAR RÉGION + SEMAINE =====
print("\n🗺️ Agrégation par région + semaine...")
print("Avant agrégation :", len(df_clean), "lignes")

# Grouper par date + région
df_regional = df_clean.groupby(['date', 'region']).agg({
    'urgences': 'mean',  # Moyenne des taux (car c'est un taux pour 100k hab)
    'sos': 'mean'
}).reset_index()

print(f"Après agrégation : {len(df_regional)} lignes")
print(f"Régions uniques : {df_regional['region'].nunique()}")
print(f"Régions : {df_regional['region'].unique()}")

# ===== SUPPRIMER NaN =====
print("\n🧹 Nettoyage des NaN...")
print(f"NaN dans urgences : {df_regional['urgences'].isnull().sum()}")
print(f"NaN dans sos : {df_regional['sos'].isnull().sum()}")

# Garder les lignes avec au moins les urgences
df_regional = df_regional[df_regional['urgences'].notna()]
print(f"Lignes après nettoyage NaN : {len(df_regional)}")

# Remplacer NaN dans sos par 0
df_regional['sos'] = df_regional['sos'].fillna(0)

# ===== TRIER =====
df_regional = df_regional.sort_values(['region', 'date']).reset_index(drop=True)

# ===== STATISTIQUES =====
print("\n📊 STATISTIQUES FINALES")
print("="*80)
print(f"Période : {df_regional['date'].min()} → {df_regional['date'].max()}")
print(f"Régions : {df_regional['region'].nunique()}")
print(f"\nUrgences :")
print(f"  Moyenne : {df_regional['urgences'].mean():.2f}")
print(f"  Min : {df_regional['urgences'].min():.2f}")
print(f"  Max : {df_regional['urgences'].max():.2f}")
print(f"\nSOS Médecins :")
print(f"  Moyenne : {df_regional['sos'].mean():.2f}")
print(f"  Min : {df_regional['sos'].min():.2f}")
print(f"  Max : {df_regional['sos'].max():.2f}")

# ===== APERÇU =====
print("\n👀 Aperçu des données :")
print(df_regional.head(10))

# ===== SAUVEGARDER =====
print("\n💾 Sauvegarde...")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
output_path = PROCESSED_DATA_DIR / 'urgences_clean.csv'
df_regional.to_csv(output_path, index=False)
print(f"✅ Fichier sauvegardé : {output_path}")

print("\n" + "="*80)
print("✅ NETTOYAGE URGENCES TERMINÉ !")
print("="*80)
