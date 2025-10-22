#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ---------- métriques "safe" ----------
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def wmape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom * 100

def smape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

# ---------- pondération des pics ----------
def make_sample_weight(y, alpha=3.0, q=95):
    """Poids entre 1 et 1+alpha. Plus y est grand (> P95), plus le poids est fort.
       Evite d'exploser les poids sur les très grands pics."""
    y = np.asarray(y, dtype=float)
    q95 = np.percentile(y, q)
    scale = max(q95, 1.0)
    w = 1.0 + alpha * (y / scale)
    return np.clip(w, 1.0, 1.0 + alpha)

# ---------- 1) Load ----------
DATA_PATH = "../../data/processed/master_dataframe.csv"
df = pd.read_csv(DATA_PATH)

# ---------- 2) Target ----------
TARGET = "urgences_grippe"
if TARGET not in df.columns:
    raise ValueError(f"Colonne '{TARGET}' absente de {DATA_PATH}")

# ---------- 3) Feature engineering ----------
df = df.sort_values(["region", "annee", "semaine_annee"])

# Lags utiles (éviter fuite d'info)
for col in ["taux_ias", "sos_medecins", "urgences_grippe"]:
    if col in df.columns:
        df[f"{col}_t-1"] = df.groupby("region")[col].shift(1)
        if col == "taux_ias":
            df[f"{col}_t-2"] = df.groupby("region")[col].shift(2)

# Features de dynamique (diff & rolling mean 3) par région
for col in ["taux_ias", "sos_medecins", "urgences_grippe"]:
    if col in df.columns:
        df[f"{col}_diff"] = df.groupby("region")[col].diff(1)
        df[f"{col}_roll3"] = df.groupby("region")[col].transform(lambda s: s.rolling(3, min_periods=1).mean())

# NEW: "accélération" épidémique (2e différence) – aide à mieux capter le début/culmination des pics
if "urgences_grippe" in df.columns:
    g = df.groupby("region")["urgences_grippe"]
    df["urgences_grippe_accel"] = g.diff(1) - g.diff(2)  # ≈ y_t - 2y_{t-1} + y_{t-2}

# Saisonnalité
df["week_sin"] = np.sin(2*np.pi*df["semaine_annee"]/52)
df["week_cos"] = np.cos(2*np.pi*df["semaine_annee"]/52)

# One-hot région
X_full = pd.get_dummies(df, columns=["region"], drop_first=True)

# Drop NaN créés par les lags/diff
needed = [
    TARGET,
    "taux_ias", "taux_ias_t-1", "taux_ias_t-2", "taux_ias_diff", "taux_ias_roll3",
    "sos_medecins", "sos_medecins_t-1", "sos_medecins_diff", "sos_medecins_roll3",
    "urgences_grippe_t-1", "urgences_grippe_diff", "urgences_grippe_roll3", "urgences_grippe_accel",
    "week_sin", "week_cos"
]
X_full = X_full.dropna(subset=[c for c in needed if c in X_full.columns])

# Features finales
region_cols = [c for c in X_full.columns if c.startswith("region_")]
features_base = [
    "taux_ias","taux_ias_t-1","taux_ias_t-2",
    "sos_medecins","sos_medecins_t-1",
    "urgences_grippe_t-1",
    "week_sin","week_cos",
    # dynamiques
    "taux_ias_diff","taux_ias_roll3",
    "sos_medecins_diff","sos_medecins_roll3",
    "urgences_grippe_diff","urgences_grippe_roll3","urgences_grippe_accel",
]
features = [c for c in features_base if c in X_full.columns] + region_cols

# X / y
y_all = X_full[TARGET].values
X_all = X_full[features].values

# ---------- 4) Backtesting (TimeSeriesSplit) avec cible log1p + sample_weight ----------
tscv = TimeSeriesSplit(n_splits=5, gap=2)
maes, rmses, r2s, mapes, wmapes, smapes = [], [], [], [], [], []

last_fold = None
model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1, min_samples_leaf=5)

y_all_log = np.log1p(y_all)  # log(1+y)
all_weights = make_sample_weight(y_all, alpha=3.0, q=95)  # pondère jusqu'à x4 les gros volumes

for i, (tr_idx, te_idx) in enumerate(tscv.split(X_all)):
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr_log, y_te = y_all_log[tr_idx], y_all[te_idx]
    w_tr = all_weights[tr_idx]

    # fit en log avec pondération des gros volumes
    model.fit(X_tr, y_tr_log, sample_weight=w_tr)
    y_hat_log = model.predict(X_te)
    y_hat = np.expm1(y_hat_log)  # retour à l'échelle réelle pour les métriques

    mae   = mean_absolute_error(y_te, y_hat)
    rmse  = mean_squared_error(y_te, y_hat, squared=False)
    r2    = r2_score(y_te, y_hat)
    mape_ = safe_mape(y_te, y_hat)
    wm    = wmape(y_te, y_hat)
    sm    = smape(y_te, y_hat)

    maes.append(mae); rmses.append(rmse); r2s.append(r2)
    mapes.append(mape_); wmapes.append(wm); smapes.append(sm)

    last_fold = (tr_idx, te_idx, y_te, y_hat)  # pour plots

# Moyennes (± std)
mae_mean, mae_std     = float(np.nanmean(maes)),  float(np.nanstd(maes))
rmse_mean, rmse_std   = float(np.nanmean(rmses)), float(np.nanstd(rmses))
r2_mean, r2_std       = float(np.nanmean(r2s)),   float(np.nanstd(r2s))
mape_mean, mape_std   = float(np.nanmean(mapes)), float(np.nanstd(mapes))
wmape_mean, wmape_std = float(np.nanmean(wmapes)),float(np.nanstd(wmapes))
smape_mean, smape_std = float(np.nanmean(smapes)),float(np.nanstd(smapes))

# Scores globaux
global_score = round(1 - (wmape_mean/100.0), 4) if not np.isnan(wmape_mean) else (
               round(1 - (mape_mean/100.0), 4) if not np.isnan(mape_mean) else np.nan)

precision_global = (
    round(((1 - wmape_mean/100.0) + r2_mean) / 2.0, 4)
    if not np.isnan(wmape_mean) and not np.isnan(r2_mean)
    else np.nan
)

print(
    "[CV 5 folds]  "
    f"MAE={mae_mean:.4f}±{mae_std:.4f} | "
    f"RMSE={rmse_mean:.4f}±{rmse_std:.4f} | "
    f"R²={r2_mean:.4f}±{r2_std:.4f} | "
    f"WMAPE={wmape_mean:.2f}%±{wmape_std:.2f}% | "
    f"sMAPE={smape_mean:.2f}%±{smape_std:.2f}% | "
    f"MAPE_safe={mape_mean:.2f}%±{mape_std:.2f}% | "
    f"Precision_globale={precision_global:.4f}"
)

# ---------- 5) Ré-entraîner sur 100% (log1p + pondération) ----------
final_weights = make_sample_weight(y_all, alpha=3.0, q=95)
model.fit(X_all, y_all_log, sample_weight=final_weights)

# ---------- 6) Save model ----------
os.makedirs("../RF_models", exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"../RF_models/urgences_rf_{stamp}.pkl"
joblib.dump(model, model_path)
print(f"✅ Modèle (100% data, log1p + pondération) sauvegardé : {model_path}")
print("ℹ️ Inférence : y_pred = np.expm1(model.predict(X_features))")

# --> Sauvegarde la liste des colonnes utilisées à l'entraînement
features_path = f"../RF_models/urgences_rf_features_{stamp}.json"
with open(features_path, "w") as f:
    json.dump(features, f)
print(f"📝 Features sauvegardées : {features_path}")

# ---------- 7) Save metrics (moyennes CV) ----------
metrics_path = "../alignment_metrics.csv"
row = {
    "model_name": "RandomForest_Urgences_CV5_log1p_weighted",
    "version": stamp,
    "mae": round(mae_mean, 4),
    "rmse": round(rmse_mean, 4),
    "r2": round(r2_mean, 4),
    "wmape": round(wmape_mean, 4) if not np.isnan(wmape_mean) else "",
    "smape": round(smape_mean, 4) if not np.isnan(smape_mean) else "",
    "mape": round(mape_mean, 4) if not np.isnan(mape_mean) else "",
    "global_score": global_score if global_score == global_score else "",
    "precision_global": precision_global if precision_global == precision_global else "",
    "timestamp": datetime.now().isoformat()
}
exists = os.path.exists(metrics_path)
pd.DataFrame([row]).to_csv(
    metrics_path, sep=",", index=False, mode="a", header=not exists
)
print(f"✅ Métriques (CV moyennes) ajoutées à : {metrics_path}")

# ---------- 8) Plots sur le dernier fold ----------
results_dir = "../results"
rf_results_dir = os.path.join(results_dir, "RF_results")
os.makedirs(rf_results_dir, exist_ok=True)

# a) Global plot (dernier fold)
_, te_idx, y_true_global, y_pred_global = last_fold
te_df = X_full.iloc[te_idx].copy()
te_df["y_true"] = y_true_global
te_df["y_pred"] = y_pred_global
te_df = te_df.sort_values(["annee", "semaine_annee"])

plt.figure(figsize=(11, 5))
plt.plot(te_df["y_true"].values, label="Réel (fold test)", linewidth=2)
plt.plot(te_df["y_pred"].values, label="Prédit (fold test)", linestyle="--")
plt.title("Urgences grippe — Réel vs Prédit (dernier fold CV)")
plt.xlabel("Observations temporelles (fold test)")
plt.ylabel("Passages urgences (grippe)")
plt.legend()
plt.tight_layout()
global_plot_path = os.path.join(rf_results_dir, f"global_reel_vs_predit_cv_last_{stamp}.png")
plt.savefig(global_plot_path, dpi=140)
plt.close()
print(f"🖼  Plot global : {global_plot_path}")

# b) Regional plots (top 4 régions) sur le dernier fold
orig_subset = df.loc[X_full.index].iloc[te_idx][["region","annee","semaine_annee", TARGET]].copy()
orig_subset["y_pred"] = y_pred_global

top_regions = orig_subset["region"].value_counts().head(4).index.tolist()
for reg in top_regions:
    g = orig_subset[orig_subset["region"] == reg].sort_values(["annee", "semaine_annee"])
    if len(g) < 5:
        continue
    plt.figure(figsize=(11, 5))
    plt.plot(g[TARGET].values, label="Réel (fold test)", linewidth=2)
    plt.plot(g["y_pred"].values, label="Prédit (fold test)", linestyle="--")
    plt.title(f"Urgences grippe — Réel vs Prédit • {reg} (fold CV)")
    plt.xlabel("Semaines (ordre temporel)")
    plt.ylabel("Passages urgences (grippe)")
    plt.legend()
    plt.tight_layout()
    reg_plot_path = os.path.join(rf_results_dir, f"{reg.replace(' ', '_')}_reel_vs_predit_cv_last_{stamp}.png")
    plt.savefig(reg_plot_path, dpi=140)
    plt.close()
    print(f"🖼  Plot région : {reg_plot_path}")
    
