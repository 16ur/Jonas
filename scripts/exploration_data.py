"""
Script 2 : Explorer les données téléchargées
On regarde ce qu'on a et on identifie les colonnes importantes
"""

print("🔍 EXPLORATION DES DONNÉES\n")

import pandas as pd
import os
from pathlib import Path

pd.set_option('display.max_columns', None)

# Trouver le répertoire racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
DOCS_DIR = BASE_DIR / 'docs'

# ===== FONCTION D'EXPLORATION =====
def explore_file(filename):
    """Explore un fichier CSV"""
    filepath = DATA_DIR / filename
    
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier non trouvé : {filename}\n")
        return None
    
    print("="*80)
    print(f"📊 FICHIER : {filename}")
    print("="*80)
    
    try:
        # Charger
        df = pd.read_csv(filepath, nrows=1000)  # Limiter pour performance
        
        print(f"\n📏 Dimensions : {len(df)} lignes × {len(df.columns)} colonnes")
        
        print(f"\n📋 Colonnes ({len(df.columns)}) :")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            print(f"   {i:2d}. {col:<50} [{dtype}] ({non_null} non-null)")
        
        print(f"\n👀 Aperçu (5 premières lignes) :")
        print(df.head())
        
        print(f"\n📈 Statistiques :")
        print(df.describe())
        
        # Chercher colonnes de date
        date_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'semaine', 'week', 'annee', 'mois'])]
        if date_cols:
            print(f"\n📅 Colonnes temporelles détectées : {date_cols}")
        
        # Chercher colonnes numériques importantes
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if num_cols:
            print(f"\n🔢 Colonnes numériques : {num_cols[:10]}")
        
        print("\n")
        return df
        
    except Exception as e:
        print(f"❌ Erreur : {e}\n")
        return None

# ===== EXPLORER LES FICHIERS PRIORITAIRES =====

print("\n" + "="*80)
print("🎯 EXPLORATION DES FICHIERS PRIORITAIRES")
print("="*80 + "\n")

# 1. Passages aux urgences (LA PLUS IMPORTANTE)
print("\n🏥 FICHIER #1 : PASSAGES AUX URGENCES + SOS MÉDECINS")
df_urgences = explore_file("urgences_sos_departement.csv")

# 2. Couvertures vaccinales
print("\n💉 FICHIER #2 : COUVERTURES VACCINALES")
df_couverture = explore_file("couverture_vaccinale_departement.csv")

# 3. Explorer UN fichier de vaccination comme exemple
print("\n📅 FICHIER #3 : VACCINATION 2024-2025 (exemple)")
vacc_files = [f for f in os.listdir(DATA_DIR) if f.startswith('vacc_2024')]
if vacc_files:
    df_vacc = explore_file(vacc_files[0])

# ===== SAUVEGARDER UN RAPPORT =====
report = []
report.append("RAPPORT D'EXPLORATION")
report.append("="*80)
report.append("")

files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
report.append(f"Nombre de fichiers : {len(files)}")
report.append("")

for f in sorted(files):
    try:
        df = pd.read_csv(DATA_DIR / f, nrows=5)
        report.append(f"\n{f}")
        report.append(f"  Lignes : {len(pd.read_csv(DATA_DIR / f))}")
        report.append(f"  Colonnes : {', '.join(df.columns[:5])}...")
    except:
        report.append(f"\n{f} - Erreur de lecture")

# Sauvegarder
os.makedirs(DOCS_DIR, exist_ok=True)
with open(DOCS_DIR / 'rapport_exploration.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("="*80)
print("✅ EXPLORATION TERMINÉE")
print("="*80)
print("\n📄 Rapport sauvegardé : docs/rapport_exploration.txt")
print("\n🎯 PROCHAINE ÉTAPE :")
print("""
ANALYSE DES COLONNES :
1. Regarde bien les colonnes ci-dessus
2. Identifie les colonnes de DATES
3. Identifie les colonnes de VALEURS (urgences, vaccinations, etc.)
4. Note les noms EXACTS pour l'étape de nettoyage

Ensuite lance : python scripts/nettoyer.py
""")
