# 🛡️ Fraudulent Insurance Claim Detection

An end-to-end supervised machine learning pipeline to classify insurance claims as **fraudulent or legitimate**, built as part of an Advanced ML case study for UpGrad (January 2025).

---

## 📌 Problem Statement

**Global Insure** processes thousands of insurance claims annually. A significant proportion are fraudulent, causing substantial financial losses. Their existing manual inspection process is slow and detects fraud too late — often after payouts have already been made.

This project builds a data-driven classification model to flag suspicious claims **early in the approval process**, minimising losses and improving operational efficiency.

---

## 🎯 Business Objectives

- Analyse historical claim data to detect patterns indicative of fraud
- Identify the most predictive features of fraudulent behaviour
- Predict fraud likelihood for incoming claims based on past data
- Extract actionable insights to improve the fraud detection process

---

## 📂 Dataset

- **Source:** `insurance_claims.csv`
- **Size:** 1,000 rows × 40 columns
- **Target variable:** `fraud_reported` (Y/N → 1/0)
- **Features include:** customer demographics, policy details, incident specifics, claim amounts

---

## 🔧 Methodology

### 1. Data Cleaning
- Dropped `_c39` (100% null)
- Imputed `authorities_contacted` nulls with `'Other'`
- Replaced `'?'` values in `collision_type`, `property_damage`, `police_report_available` with `'Unknown'`
- Corrected negative `umbrella_limit` to mode (0)
- Removed high-cardinality identifier columns: `policy_number`, `policy_bind_date`, `insured_zip`, `incident_location`

### 2. Train-Validation Split
- 70/30 stratified split on `fraud_reported`
- Training: 700 samples | Validation: 300 samples

### 3. Exploratory Data Analysis (EDA)
- **Univariate:** Histograms for all numerical features
- **Correlation:** Heatmap to identify multicollinearity
- **Class balance:** Confirmed imbalance (~75% legitimate, ~25% fraud)
- **Bivariate:** Bar plots (categorical vs. fraud rate), box plots (numerical vs. fraud)

### 4. Feature Engineering
- **Resampling:** `RandomOverSampler` to balance training classes (527 → 527 each)
- **Feature creation:** `incident_month` (from date), `auto_age` (from `auto_year`)
- **Dropped:** `age` (high correlation with `months_as_customer`), `total_claim_amount` (sum of sub-claims), `incident_date` (after extraction)
- **Binning:** `incident_hour_of_the_day` → `incident_period_of_the_day` (Night/Morning/Afternoon/Evening)
- **Encoding:** One-hot encoding with `drop_first=True`
- **Scaling:** `StandardScaler` on all numerical features (fit on train, transform on validation)

### 5. Model Building

#### Logistic Regression
- Feature selection via **RFECV** (optimal: 102 features)
- Model built using `statsmodels.GLM` (Binomial family)
- Multicollinearity assessed using **VIF**; high-VIF features removed
- Optimal cutoff determined via ROC, accuracy-sensitivity-specificity, and precision-recall curves

#### Random Forest
- Baseline `RandomForestClassifier` → 100% train accuracy (overfitting)
- Cross-validation mean accuracy: **92.13%**
- Feature importance filtering (threshold ≥ 0.02) → **7 features retained**
- Hyperparameter tuning via **GridSearchCV** (756 candidate combinations, 5-fold CV)
- Best params: `max_depth=10`, `max_features=8`, `min_samples_leaf=10`, `min_samples_split=20`, `n_estimators=15`

---

## 📊 Results

### Validation Set Performance

| Metric | Logistic Regression | Random Forest (Tuned) |
|---|---|---|
| Accuracy | 80.33% | **82.00%** |
| Sensitivity (Recall) | 62.16% | **75.68%** |
| Specificity | 86.28% | 84.07% |
| Precision | 59.74% | 60.87% |
| F1-Score | 0.609 | **0.675** |

### Key Finding: Top Predictive Features (Random Forest)

| Feature | Importance |
|---|---|
| `incident_severity_Minor Damage` | Highest |
| `vehicle_claim` | High |
| `insured_hobbies_chess` | Medium-High |
| `incident_severity_Total Loss` | Medium-High |
| `capital-loss` | Medium |
| `insured_hobbies_cross-fit` | Medium |
| `witnesses` | Medium |

---

## 🏆 Conclusion

The **Tuned Random Forest** model outperformed Logistic Regression on the validation set, achieving **82% accuracy** and an **F1-score of 0.675** with significantly higher sensitivity (75.68% vs. 62.16%). This makes it substantially better at actually catching fraudulent claims — the primary business objective.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Manipulation | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| ML Models | scikit-learn |
| Statistical Modelling | statsmodels |
| Class Imbalance | imbalanced-learn (RandomOverSampler) |
| Feature Selection | RFECV |
| Hyperparameter Tuning | GridSearchCV |

---

## 📁 Project Structure

```
├── Fraudulent_Claim_Detection_Syed_Ismail_N.ipynb   # Full notebook
├── insurance_claims.csv                              # Dataset (not included)
└── README.md
```

---


- Syed Ismail N


*Advanced ML Case Study — UpGrad, January 2026*
