# 🏦 Loan Default Prediction

> End-to-end credit risk pipeline — Give Me Some Credit (Kaggle)

---

## 📌 Project Overview

This project predicts whether a borrower will experience serious financial distress within 2 years, using a full data science pipeline from raw data to a deployed Streamlit app.

**Course:** Fundamentals of Data Science — Third Year  
**Faculty:** Artificial Intelligence, Menoufia University  
**Dataset:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## 📁 Project Structure

```
Data Science complete project/
├── 4_5791903174071688749.ipynb       # Main analysis notebook
├── app.py                            # Streamlit prediction GUI
├── cs-training.csv                   # Training data (~150K rows)
├── cs-test.csv                       # Test data
├── Data Dictionary.xls               # Feature descriptions
├── model_artifacts/
│   ├── best_model.pkl                # Trained LightGBM model
│   ├── best_threshold.pkl            # Optimized decision threshold
│   ├── feature_columns.pkl           # Feature order for inference
│   └── model_info.pkl                # Model metadata
└── roc_pr_all_models.png             # ROC & PR curve comparison
```

---

## 🗃️ Dataset

| Property | Value |
|----------|-------|
| Training set | ~150,000 rows |
| Features | 10 behavioral/financial features |
| Target | `SeriousDelinquency` (1 = default within 2 years) |
| Class imbalance | ~6.7% defaults vs ~93.3% non-defaults |

---

## ⚙️ Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Data Collection |
| 2 | Data Cleaning (age=0, sentinel codes 96/98, Winsorization) |
| 3 | Feature Engineering (debt ratios, utilization patterns) |
| 4 | Feature Selection (SelectKBest + Recursive Feature Elimination) |
| 5A | Imbalance Handling — SMOTE (oversampling) |
| 5B | Imbalance Handling — Class Weights |
| 5C | Full Model Comparison & Best Selection |
| 6 | Visualization (EDA, model performance, SHAP-style analysis) |

---

## 🤖 Models Compared

| Model | Approach |
|-------|----------|
| Logistic Regression | Baseline |
| Random Forest | SMOTE + Class Weights |
| LightGBM ✅ | **Best model — selected** |
| Stacking Ensemble | Meta-learner on top |

**Selection metric:** AUC-PR (more reliable than ROC-AUC for imbalanced data)

---

## 🎯 Key Techniques

- **SMOTE-ENN** — combined over/under sampling for cleaner boundaries
- **Optuna** — 100-trial hyperparameter optimization for LightGBM
- **Cost-sensitive threshold optimization** — maximize business utility (not just accuracy)
- **Stacking ensemble** — Logistic Regression meta-learner on base model predictions

---

## 🖥️ Streamlit App

The `app.py` supports two prediction modes:

**Single Customer Mode:** fill in financial features → get risk score + recommendation  
**Batch Mode:** upload CSV → download results with risk flags

**Run:**
```bash
streamlit run app.py
```

> If `model_artifacts/` is missing, the app auto-trains a lightweight demo model on synthetic data.

---

## 🛠️ Tech Stack

`Python` · `LightGBM` · `Scikit-learn` · `Imbalanced-learn` · `Optuna` · `Pandas` · `Streamlit` · `Joblib`
