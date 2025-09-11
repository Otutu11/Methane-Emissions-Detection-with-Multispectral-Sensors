# file: methane_emissions_multispectral_demo.py
# Purpose: End-to-end synthetic demo for "Methane-Emissions-Detection-with-Multispectral-Sensors"
# Author: You :)
# Run: python methane_emissions_multispectral_demo.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from joblib import dump

# -----------------------------
# 0) Setup
# -----------------------------
np.random.seed(42)
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# 1) Generate synthetic data (>100)
# -----------------------------
N = 1500  # plenty more than 100

# Multispectral reflectances (0..1)
blue  = np.clip(np.random.normal(0.12, 0.03, N), 0, 1)
green = np.clip(np.random.normal(0.18, 0.04, N), 0, 1)
red   = np.clip(np.random.normal(0.20, 0.05, N), 0, 1)
nir   = np.clip(np.random.normal(0.40, 0.08, N), 0, 1)
swir1 = np.clip(np.random.normal(0.26, 0.06, N), 0, 1)   # ~1.6 μm
swir2 = np.clip(np.random.normal(0.22, 0.06, N), 0, 1)   # ~2.2 μm (CH4-sensitive proxy)

# Thermal + scene/atmos
tir_bt           = np.clip(np.random.normal(300, 6,  N), 250, 330)  # K
surface_pressure = np.random.normal(1013, 8, N)                     # hPa
humidity         = np.clip(np.random.normal(0.55, 0.15, N), 0, 1)   # 0..1
wind_speed       = np.clip(np.random.gamma(2.0, 1.5, N), 0, 20)     # m/s
solar_zenith     = np.clip(np.random.normal(35, 12, N), 0, 80)      # deg

# Leak presence and intensity
leak = np.random.binomial(1, 0.35, N)  # 35% of scenes contain a leak
base_ch4 = np.random.normal(2.0, 0.3, N)  # ppm-eq background enhancement
leak_strength = leak * np.random.gamma(shape=2.0, scale=1.2, size=N)  # emission intensity

# Spectral effects of methane (lower SWIR reflectance with more CH4)
swir2_effect = -0.25 * leak_strength + np.random.normal(0, 0.02, N)
swir2_obs = np.clip(swir2 + swir2_effect, 0, 1)

swir1_effect = -0.08 * leak_strength + np.random.normal(0, 0.015, N)
swir1_obs = np.clip(swir1 + swir1_effect, 0, 1)

# Continuous target: methane column enhancement (ppm-eq)
ch4_ppm_eq = (
    base_ch4
    + 0.9 * leak_strength
    + 0.02 * (tir_bt - 295)        # warmer scenes -> more buoyant plumes
    - 0.05 * (wind_speed - 4)      # high wind disperses signal
    + 0.3  * (0.25 - swir2_obs)    # optical absorption cue
    + np.random.normal(0, 0.15, N)
)

# Binary detection label: top 40% CH4 enhancement = emission present
thresh = np.percentile(ch4_ppm_eq, 60)
emission_label = (ch4_ppm_eq >= thresh).astype(int)

# Derived indices
ndvi   = (nir - red) / (nir + red + 1e-6)
ndsi   = (swir1_obs - swir2_obs) / (swir1_obs + swir2_obs + 1e-6)   # SWIR ratio
nbr    = (nir - swir2_obs) / (nir + swir2_obs + 1e-6)
albedo = 0.1*blue + 0.1*green + 0.2*red + 0.3*nir + 0.15*swir1_obs + 0.15*swir2_obs

df = pd.DataFrame({
    "blue": blue, "green": green, "red": red, "nir": nir,
    "swir1": swir1_obs, "swir2": swir2_obs, "tir_bt": tir_bt,
    "surface_pressure": surface_pressure, "humidity": humidity,
    "wind_speed": wind_speed, "solar_zenith": solar_zenith,
    "ndvi": ndvi, "ndsi": ndsi, "nbr": nbr, "albedo": albedo,
    "ch4_ppm_eq": ch4_ppm_eq, "emission_label": emission_label
})

csv_path = os.path.join(OUTDIR, "synthetic_methane_multispectral.csv")
df.to_csv(csv_path, index=False)

# -----------------------------
# 2) Split
# -----------------------------
features = [
    "blue","green","red","nir","swir1","swir2","tir_bt",
    "surface_pressure","humidity","wind_speed","solar_zenith",
    "ndvi","ndsi","nbr","albedo"
]
X = df[features].values
y_reg = df["ch4_ppm_eq"].values
y_cls = df["emission_label"].values

Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

# -----------------------------
# 3) Regression model
# -----------------------------
reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
reg.fit(Xtr_r, ytr_r)
yp_r = reg.predict(Xte_r)

mae  = mean_absolute_error(yte_r, yp_r)
rmse = mean_squared_error(yte_r, yp_r, squared=False)
r2   = r2_score(yte_r, yp_r)

# Save parity plot
plt.figure()
plt.scatter(yte_r, yp_r, alpha=0.6)
mn = min(yte_r.min(), yp_r.min()); mx = max(yte_r.max(), yp_r.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True CH4 enhancement (ppm-eq)")
plt.ylabel("Predicted CH4 enhancement (ppm-eq)")
plt.title(f"Regression Parity | MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "regression_parity.png")); plt.close()

# Feature importance (regression)
imp_r = pd.Series(reg.feature_importances_, index=features).sort_values()
plt.figure()
plt.barh(imp_r.index, imp_r.values)
plt.xlabel("Importance")
plt.title("Regression Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "regression_feature_importance.png")); plt.close()

# -----------------------------
# 4) Classification model
# -----------------------------
clf = GradientBoostingClassifier(random_state=42)
clf.fit(Xtr_c, ytr_c)
proba = clf.predict_proba(Xte_c)[:, 1]
yp_c  = (proba >= 0.5).astype(int)

acc = (yp_c == yte_c).mean()
auc = roc_auc_score(yte_c, proba)

# ROC
fpr, tpr, thr = roc_curve(yte_c, proba)
plt.figure()
plt.plot(fpr, tpr); plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC={auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "classification_roc.png")); plt.close()

# Confusion Matrix
cm = confusion_matrix(yte_c, yp_c)
plt.figure()
im = plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks([0,1], ["No Emission","Emission"])
plt.yticks([0,1], ["No Emission","Emission"])
for (i,j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha="center", va="center")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "classification_confusion.png")); plt.close()

# Report
report = classification_report(yte_c, yp_c, target_names=["No Emission","Emission"])
with open(os.path.join(OUTDIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Surrogate RF for feature importance (interpretability)
rf_sur = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_sur.fit(Xtr_c, ytr_c)
imp_c = pd.Series(rf_sur.feature_importances_, index=features).sort_values()
plt.figure()
plt.barh(imp_c.index, imp_c.values)
plt.xlabel("Importance")
plt.title("Classification Feature Importances (RF surrogate)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "classification_feature_importance.png")); plt.close()

# Cross-validation (AUC)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(clf, X, y_cls, cv=cv, scoring="roc_auc")
with open(os.path.join(OUTDIR, "cv_results.txt"), "w") as f:
    f.write(f"ROC-AUC (5-fold): mean={cv_auc.mean():.3f}, std={cv_auc.std():.3f}\n")
    f.write(f"Fold scores: {np.round(cv_auc, 3)}\n")

# -----------------------------
# 5) Save artifacts
# -----------------------------
dump(reg, os.path.join(OUTDIR, "methane_regressor.joblib"))
dump(clf, os.path.join(OUTDIR, "methane_classifier.joblib"))

with open(os.path.join(OUTDIR, "README_Methane_Emissions_Demo.txt"), "w") as f:
    f.write(
        "Methane-Emissions-Detection-with-Multispectral-Sensors (Synthetic Demo)\n\n"
        "Files:\n"
        "- synthetic_methane_multispectral.csv : synthetic dataset\n"
        "- methane_regressor.joblib : RandomForestRegressor for CH4 enhancement (ppm-eq)\n"
        "- methane_classifier.joblib : GradientBoostingClassifier for emission detection\n"
        "- regression_parity.png, regression_feature_importance.png\n"
        "- classification_roc.png, classification_confusion.png, classification_feature_importance.png\n"
        "- classification_report.txt, cv_results.txt\n\n"
        "Quick usage:\n"
        "from joblib import load\n"
        "import pandas as pd\n"
        "df = pd.read_csv('outputs/synthetic_methane_multispectral.csv')\n"
        "X = df[['blue','green','red','nir','swir1','swir2','tir_bt','surface_pressure',"
        "'humidity','wind_speed','solar_zenith','ndvi','ndsi','nbr','albedo']].values\n"
        "reg = load('outputs/methane_regressor.joblib'); y_ch4 = reg.predict(X)\n"
        "clf = load('outputs/methane_classifier.joblib'); p_leak = clf.predict_proba(X)[:,1]\n"
    )

# -----------------------------
# 6) Print summary to console
# -----------------------------
print("=== Regression ===")
print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
print("\n=== Classification ===")
print(f"Accuracy={acc:.3f}  ROC-AUC={auc:.3f}")
print("\nClassification report:\n", report)
print(f"\n5-fold ROC-AUC: mean={cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
print(f"\nArtifacts saved to: {OUTDIR}/")
