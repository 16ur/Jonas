"""
Script 1 : Télécharger TOUTES les données vaccination grippe
Ce script télécharge les données depuis data.gouv.fr et Santé Publique France
"""

print("🚀 Début du téléchargement des données...\n")

import pandas as pd
import os
import requests
from time import sleep

# Créer les dossiers
os.makedirs('../data/raw', exist_ok=True)

# ===== FONCTION DE TÉLÉCHARGEMENT =====
def download_file(url, filename, description):
    """Télécharge un fichier CSV"""
    try:
        print(f"📥 Téléchargement : {description}...")

        # Télécharger
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Sauvegarder
        output_path = f'../data/raw/{filename}'

        # Si c'est un CSV directement
        if url.endswith('.csv') or 'exports/csv' in url:
            with open(output_path, 'wb') as f:
                f.write(response.content)
        else:
            # Sinon, charger avec pandas et sauvegarder
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), sep=None, engine='python')
            df.to_csv(output_path, index=False)

        # Vérifier
        df_check = pd.read_csv(output_path, nrows=5)
        print(f"   ✅ OK - {len(pd.read_csv(output_path))} lignes")
        print(f"   📁 Sauvegardé : {output_path}\n")

        return True

    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        print(f"   💡 Télécharge manuellement depuis : {url}\n")
        return False

# ===== DONNÉES PRIORITAIRES POUR LE HACKATHON =====

print("="*70)
print("🎯 TÉLÉCHARGEMENT DES DONNÉES ESSENTIELLES")
print("="*70 + "\n")

# 1. PASSAGES AUX URGENCES (LA PLUS IMPORTANTE - notre cible)
print("🏥 1/2 - PASSAGES AUX URGENCES + SOS MÉDECINS\n")
urgences_url = "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/grippe-passages-aux-urgences-et-actes-sos-medecins-departement/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%2C"
download_file(urgences_url, "urgences_sos_departement.csv", "Passages urgences + SOS Médecins")

# 2. COUVERTURES VACCINALES (contexte)
print("💉 2/2 - COUVERTURES VACCINALES\n")
couverture_url = "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/couvertures-vaccinales-des-adolescent-et-adultes-departement/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%2C"
download_file(couverture_url, "couverture_vaccinale_departement.csv", "Couvertures vaccinales")

print("\n" + "="*70)
print("📊 TÉLÉCHARGEMENT DES DONNÉES VACCINATION (HISTORIQUE)")
print("="*70 + "\n")

# ===== DONNÉES VACCINATION PAR ANNÉE =====
# Ces données viennent de data.gouv.fr
# Format URL : https://www.data.gouv.fr/fr/datasets/r/{resource_id}

vaccination_data = {
    # 2024-2025
    "2024-2025": [
        ("848e3e48-4971-4dc5-97c7-d856cdfde2f6", "vacc_2024_2025_1.csv"),
        ("c26a4606-b6a5-49a9-ad18-89cc0e1fc8c2", "vacc_2024_2025_2.csv"),
        ("a6f6c78f-96eb-41e5-8c8e-54c2bcbbe3f2", "vacc_2024_2025_3.csv"),
    ],
    # 2023-2024
    "2023-2024": [
        ("cd098e63-9fba-4e83-80c7-a6bc6d5bb015", "vacc_2023_2024_1.csv"),
        ("c8fb9309-5938-43a1-8567-614d44c34c01", "vacc_2023_2024_2.csv"),
        ("ffb8591b-5fa1-4071-9d2c-640ba873f798", "vacc_2023_2024_3.csv"),
    ],
    # 2022-2023
    "2022-2023": [
        ("c36e85ef-077e-465e-b3e4-e218d972f45e", "vacc_2022_2023_1.csv"),
        ("992e690a-0c9c-4457-a556-3d70b4af29e8", "vacc_2022_2023_2.csv"),
        ("1339c744-6b09-4a21-a5a0-6f56200f7208", "vacc_2022_2023_3.csv"),
    ],
    # 2021-2022
    "2021-2022": [
        ("70f1cfba-569c-46fd-aa0d-2bc890a42eb5", "vacc_2021_2022_1.csv"),
        ("b4867c67-70c9-459f-a88d-859996e8098b", "vacc_2021_2022_2.csv"),
        ("d1a7a9c8-da2a-4840-be4e-720b4703462c", "vacc_2021_2022_3.csv"),
    ],
}

# Télécharger toutes les données de vaccination
for year, files in vaccination_data.items():
    print(f"\n📅 ANNÉE {year}")
    for resource_id, filename in files:
        url = f"https://www.data.gouv.fr/fr/datasets/r/{resource_id}"
        download_file(url, filename, f"Vaccination {year} - {filename}")
        sleep(0.5)  # Pause pour ne pas surcharger le serveur

# ===== RÉSUMÉ =====
print("\n" + "="*70)
print("✅ TÉLÉCHARGEMENT TERMINÉ")
print("="*70)

# Lister les fichiers téléchargés
downloaded = []
raw_dir = '../data/raw'
if os.path.exists(raw_dir):
    files = os.listdir(raw_dir)
    downloaded = [f for f in files if f.endswith('.csv')]

print(f"\n📊 {len(downloaded)} fichiers téléchargés dans data/raw/")
print("\n📂 Fichiers disponibles :")
for f in sorted(downloaded):
    size = os.path.getsize(os.path.join(raw_dir, f)) / 1024  # Ko
    print(f"   • {f:<40} ({size:,.1f} Ko)")

print("\n" + "="*70)
print("🎯 PROCHAINE ÉTAPE")
print("="*70)
print("""
Maintenant, explore les données téléchargées :

    python scripts/explorer.py

Les fichiers PRIORITAIRES pour ton modèle :
  1. urgences_sos_departement.csv → CIBLE (ce qu'on veut prédire)
  2. couverture_vaccinale_departement.csv → CONTEXTE
  3. vacc_YYYY_*.csv → DONNÉES VACCINATION (optionnel)

⚠️ NOTE : Ces liens ne contiennent PAS l'IAS (Indicateur Avancé Sanitaire).
L'IAS semble ne plus être disponible publiquement.
On va utiliser les données de VACCINATION comme proxy !
""")
