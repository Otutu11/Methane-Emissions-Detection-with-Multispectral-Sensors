# Methane Emissions Detection with Multispectral Sensors

This repository demonstrates how **artificial intelligence (AI)** and **multispectral remote sensing** can be applied to detect and quantify methane (CH₄) emissions.  
Using a **synthetic dataset** (simulated reflectance bands, thermal channels, and atmospheric variables), we train:

- A **regression model** to estimate methane column enhancement (ppm-eq).  
- A **classification model** to detect the presence of methane leaks.  

The pipeline integrates data generation, preprocessing, model training, evaluation, and visualization.

---

## 🚀 Features
- Synthetic **multispectral dataset** (>1,500 samples) with realistic CH₄-driven spectral effects.
- Derived vegetation and spectral indices (NDVI, NDSI, NBR, Albedo).
- **Random Forest Regressor** for CH₄ enhancement prediction.
- **Gradient Boosting Classifier** for emission detection.
- Evaluation metrics: MAE, RMSE, R² (regression), Accuracy, ROC-AUC, Confusion Matrix (classification).
- Visual outputs: regression parity plots, ROC curves, feature importances, confusion matrices.

---

## 📂 Project Structure
.
├── outputs/
│ ├── synthetic_methane_multispectral.csv
│ ├── methane_regressor.joblib
│ ├── methane_classifier.joblib
│ ├── regression_parity.png
│ ├── regression_feature_importance.png
│ ├── classification_roc.png
│ ├── classification_confusion.png
│ ├── classification_feature_importance.png
│ ├── classification_report.txt
│ ├── cv_results.txt
│ └── README_Methane_Emissions_Demo.txt
├── methane_emissions_multispectral_demo.py
└── README.md

yaml
Copy code

---

## 📊 Dataset
The dataset is synthetically generated to mimic real-world methane absorption and scene variability.  
It includes:

- **Bands**: Blue, Green, Red, NIR, SWIR1, SWIR2, TIR  
- **Scene Variables**: Surface pressure, humidity, wind speed, solar zenith  
- **Indices**: NDVI, NDSI, NBR, Albedo  
- **Targets**:  
  - `ch4_ppm_eq`: methane column enhancement (continuous, ppm-eq)  
  - `emission_label`: binary indicator (1 = emission present, 0 = none)  

---

## 🧪 Methodology
1. **Synthetic Data Generation**  
   - Simulates spectral absorption in SWIR bands due to methane plumes.  
   - Includes meteorological drivers (wind, temperature) affecting detection.  

2. **Feature Engineering**  
   - Computation of vegetation and reflectance indices for sensitivity.  

3. **Model Training**  
   - Regression → RandomForestRegressor.  
   - Classification → GradientBoostingClassifier.  

4. **Evaluation**  
   - Regression: MAE, RMSE, R² + Parity plots.  
   - Classification: Accuracy, ROC-AUC, Confusion Matrix, Feature Importances.  

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/Methane-Emissions-Detection-with-Multispectral-Sensors.git
cd Methane-Emissions-Detection-with-Multispectral-Sensors
pip install -r requirements.txt
requirements.txt

nginx
Copy code
numpy
pandas
scikit-learn
matplotlib
joblib
▶️ Usage
Run the demo script:

bash
Copy code
python methane_emissions_multispectral_demo.py
This will generate:

A synthetic dataset (outputs/synthetic_methane_multispectral.csv)

Trained models (.joblib)

Evaluation plots + reports in the outputs/ folder

Quick Inference
python
Copy code
from joblib import load
import pandas as pd

# Load dataset
df = pd.read_csv("outputs/synthetic_methane_multispectral.csv")
X = df[['blue','green','red','nir','swir1','swir2','tir_bt','surface_pressure',
        'humidity','wind_speed','solar_zenith','ndvi','ndsi','nbr','albedo']].values

# Load models
reg = load("outputs/methane_regressor.joblib")
clf = load("outputs/methane_classifier.joblib")

# Predict
y_ch4 = reg.predict(X)             # ppm-eq methane estimate
p_leak = clf.predict_proba(X)[:,1] # probability of leak
📈 Results (Synthetic Example)
Regression

MAE = ~0.40 ppm-eq

RMSE = ~0.57 ppm-eq

R² = ~0.85

Classification

Accuracy = ~89%

ROC-AUC = ~0.93

Cross-validated ROC-AUC = ~0.92 ± 0.01

🌍 Applications
Methane leak detection in oil & gas facilities.

Greenhouse gas monitoring from airborne/satellite sensors.

Climate change mitigation & compliance monitoring.

📚 References
Cusworth, D. H., et al. (2021). "A review of satellite remote sensing for methane emissions detection." Atmospheric Environment.

Thorpe, A. K., et al. (2017). "Mapping methane concentrations from airborne remote sensing." Remote Sensing of Environment.

Jongaramrungruang, S., et al. (2019). "Towards accurate methane detection using hyperspectral imaging." Atmospheric Measurement Techniques.

📝 License
This project is released under the MIT License.

Author: Anslem Otutu
Github: @Otutu11
