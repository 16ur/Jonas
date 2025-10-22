#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "../RF_models/urgences_rf_20251022_111917.pkl"           
FEATURES_JSON = "../RF_models/urgences_rf_features_20251022_111917.json"             
DATA_PATH = "../../data/processed/master_dataframe.csv"

# --- 1) Load modèle + features attendues ---
model = joblib.load(MODEL_PATH)
with open(FEATURES_JSON, "r") as f:
    feature_cols = json.load(f)
print(f"✅ Modèle chargé\n✅ {len(feature_cols)} features attendues")

# --- 2) Charger la data brute ---
df = pd.read_csv(DATA_PATH)

# --- 3) REPRODUIRE LE MÊME PREPROCESS QUE PENDANT LE TRAIN ---
df = df.sort_values(["region", "annee", "semaine_annee"])

# Lags
for col in ["taux_ias", "sos_medecins", "urgences_grippe"]:
    if col in df.columns:
        df[f"{col}_t-1"] = df.groupby("region")[col].shift(1)
        if col == "taux_ias":
            df[f"{col}_t-2"] = df.groupby("region")[col].shift(2)

# Dynamiques (diff + rolling 3)
for col in ["taux_ias", "sos_medecins", "urgences_grippe"]:
    if col in df.columns:
        df[f"{col}_diff"] = df.groupby("region")[col].diff(1)
        df[f"{col}_roll3"] = df.groupby("region")[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )

# Accélération épidémique (2e diff)
if "urgences_grippe" in df.columns:
    g = df.groupby("region")["urgences_grippe"]
    df["urgences_grippe_accel"] = g.diff(1) - g.diff(2)

# Saison
df["week_sin"] = np.sin(2*np.pi*df["semaine_annee"]/52)
df["week_cos"] = np.cos(2*np.pi*df["semaine_annee"]/52)

# One-hot région
X_full = pd.get_dummies(df, columns=["region"], drop_first=True)

# Drop NaN lags/diffs (comme au train)
needed = [
    "taux_ias","taux_ias_t-1","taux_ias_t-2","taux_ias_diff","taux_ias_roll3",
    "sos_medecins","sos_medecins_t-1","sos_medecins_diff","sos_medecins_roll3",
    "urgences_grippe_t-1","urgences_grippe_diff","urgences_grippe_roll3","urgences_grippe_accel",
    "week_sin","week_cos"
]
X_full = X_full.dropna(subset=[c for c in needed if c in X_full.columns])

# --- 4) Aligner les colonnes EXACTEMENT comme au training ---
for c in feature_cols:
    if c not in X_full:
        X_full[c] = 0
X_features = X_full[feature_cols]   # ordre strict

# --- 5) Prédire (modèle entraîné sur log1p) ---
y_pred_log = model.predict(X_features.head(10))      # exemple: 10 premières lignes valides
y_pred_real = np.expm1(y_pred_log)                   # retour à l’échelle réelle

print("✅ échantillon de prédictions (échelle réelle) :", np.round(y_pred_real, 1))
print("ℹ️ nb de lignes prêtes à l’inférence :", len(X_features))