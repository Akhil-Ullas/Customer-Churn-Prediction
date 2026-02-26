
# Customer Churn Prediction

A machine learning and deep learning project to predict customer churn for a telecommunications company. Completed with a Streamlit web application for real-time predictions and a professional bashboard for stake holders.


## Overview

Customer churn is one of the most costly problems in the telecom industry. This project builds both a classical machine learning pipeline and a deep learning model (ANN) to identify customers at risk of canceling their subscription. The final model is deployed as an interactive **Streamlit web** that allows real-time churn predictions with risk level classification.

---

## Dataset

**Source:** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

You can find the detailed discription of the dataset in the above link

| Property | Detail |
|---|---|
| Records | 7,043 customers |
| Features | 20 columns |
| Target | `Churn` (Yes / No) |

---

## Key Findings

> Insights discovered during EDA

* Customers who choose Month-to-Month contract are most likely to churn
* Customers with Fiber Optic internet services are likely to churn
* Customers who uses electronic check as payment method have high risk of churn 
* Customers who don't have online security are more likely to churn

---

## Correalation of features with Churn

* gender           : -0.0086 -> No predictive value (can be dropped)
* seniorcitizen    :  0.15   -> weak
* partner          : -0.15   -> strong
* Dependents       : -0.16   -> strong
* tenure           : -0.15   -> strong
* phoneservice     : 0.012   -> No predictive value (can be dropped)
* Multiplelines    : 0.038   -> No predictive value (can be dropped)
* internetservices : -0.047  -> moderate
* onlinesecurity   : -0.29   -> strong
* onlinebackup     : -0.2    -> strong
* Deviceprotection : -0.18   -> strong
* Techsupport      : -0.28   -> strong
* streamingTV      : -0.037  -> weak
* contract         : -0.4    -> strong
* paperlessbilling : 0.19    -> weak
* paymentmethod    : 0.11    -> weak
* Monthlycharges   : 0.19    -> weak
* totalcharges     : 0.014   -> No predictive value
---

##  Workflow

```
Data Loading & EDA
       ↓
Feature Encoding (Label Encoding)
       ↓
Correlation Analysis & Feature Selection
       ↓
Train/Test Split (80/20)
       ↓
Class Balancing (SMOTE)
       ↓
─────────────────────────────────────────
      ML Pipeline          DL Pipeline
─────────────────────────────────────────
 Model Benchmarking      StandardScaler
  (11 classifiers)             ↓
         ↓               SMOTE Balancing
  Hyperparameter               ↓
  Tuning (GridSearchCV)   ANN (Keras)
         ↓               64→32→16→1
  Threshold Opt.         Dropout + Adam
      (0.3)              EarlyStopping
─────────────────────────────────────────
                ↓
  Final Model Export (.pkl / .keras)
                ↓
       Power BI Integration
                ↓
       Streamlit Web App 
```

---

##  ML Models & Results

### Cross-Validation Accuracy (SMOTE balanced)

| Model | CV Accuracy |
|---|---|
|  Random Forest | 83.99% |
|  LightGBM | 83.58% |
|  XGBoost | 83.46% |
|  CatBoost | 83.35% |
|  Gradient Boosting | 81.91% |
| AdaBoost | 80.50% |
| Logistic Regression | 79.42% |
| Decision Tree | 78.02% |
| BernoulliNB | 76.78% |
| KNN | 76.53% |
| SVM | 57.08% |

### After Hyperparameter Tuning (GridSearchCV)

| Model | Best Score | Best Params |
|---|---|---|
| Random Forest | 84.15% | `n_estimators=200, max_depth=20` |
| CatBoost | 83.91% | `depth=8, iterations=500, learning_rate=0.05` |
| LightGBM | 83.70% | `n_estimators=500, learning_rater=0.05` |
| XGBoost | 83.51% | `n_estimators=150, max_depth=6, learning_rate=0.1` |
| GradientBoost | 83.29% | `n_estimators=300, max_depth=5, learning_rate=0.1` |

###  Selected ML Model : LightGBM 

**Threshold adjusted to 0.3** to maximize recall on the churn class.

| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.90 | 0.52 |
| Recall | 0.75 | **0.78** |
| F1-Score | 0.82 | 0.63 |
| Accuracy | | **75%** |
| **AUC** | | **0.834** |

---

##  Deep Learning Model (ANN)

A separate ANN was built using TensorFlow/Keras on the same preprocessed and SMOTE-balanced data, with StandardScaler applied before training.

### Architecture

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Dense (64, ReLU)              │ (None, 64)             │         1,280 │
├───────────────────────────────┼────────────────────────┼───────────────┤
│ Dense (32, ReLU)              │ (None, 32)             │         2,080 │
├───────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout (0.3)                 │ (None, 32)             │             0 │
├───────────────────────────────┼────────────────────────┼───────────────┤
│ Dense (16, ReLU)              │ (None, 16)             │           528 │
├───────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout (0.2)                 │ (None, 16)             │             0 │
├───────────────────────────────┼────────────────────────┼───────────────┤
│ Dense (1, Sigmoid)            │ (None, 1)              │            17 │
└───────────────────────────────┴────────────────────────┴───────────────┘
                                          Total Params:   3,905
```

### Training Configuration

| Setting | Detail |
|---|---|
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Regularization | Dropout (0.3, 0.2) |
| Weight Init | He Uniform |
| Overfitting Control | EarlyStopping (patience=5, monitor=val_loss) |
| Decision Threshold | 0.3 |
| Saved As | `deep_learning_model.keras` |

### ANN Test Set Performance

| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.89 | 0.49 |
| Recall | 0.71 | **0.76** |
| F1-Score | 0.79 | 0.60 |
| Accuracy | | **73%** |
| **AUC** | | **0.812** |

---

##  Model Comparison

| Model | Accuracy | AUC | Churn Recall | Churn Precision |
|---|---|---|---|---|
|  LightGBM | 75% | **0.834** | **0.78** | 0.52 |
|  ANN | 73% | 0.812 | 0.76 | 0.49 |

> Both models use a **0.3 decision threshold** to prioritize catching churners over precision.
> LightGBM edges out the ANN on all metrics while being significantly faster to train, making it the recommended production model.

---

##  Streamlit Web

The final LightGBM model is deployed as an interactive web application built with Streamlit, allowing business users to get real-time churn predictions without any coding.

### Features

-  **Input Form** : Enter all 19 customer attributes via dropdowns and number inputs
-  **Churn Probability** : Displays the model's predicted probability score
-  **Risk Level Classification:**

| Risk Level | Probability Range | Indicator |
|---|---|---|
|  Low Risk | < 0.3 | 
|  Medium Risk | 0.3 – 0.6 | 
|  High Risk | > 0.6 | 

### How It Works

```
User fills in customer details (19 features)
              ↓
   Input encoded via pd.get_dummies()
              ↓
  Features aligned to training columns
              ↓
  LightGBM predicts churn probability
              ↓
  Threshold 0.3 applied → Churn / No Churn
              ↓
     Risk level displayed to user
```

### Running the Website

```bash
streamlit run deployment.py
```

---


## Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
tensorflow
keras
streamlit
joblib
jupyter
```


---

##  How to Run

This project was developed using **GitHub Codespaces**

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/Akhil-Ullas/Customer-Churn-Prediction)

```bash
# 1. Open the repository in GitHub Codespaces
#    Click the green "Code" button → "Codespaces" tab → "Create codespace on main"

# 2. Add the dataset
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv inside the /Data folder (Raw dataset)

# 3. Run ML notebook
jupyter notebook main.ipynb

# 4. Run Deep Learning notebook
jupyter notebook deep_learning.ipynb

# 5. Launch Streamlit App
streamlit run deployment.py
```
---

## Project Structure
```
CUSTOMER-CHURN-PREDICTION/
│
|── Dashboard/
│   ├── Preview/
|   |       ├── Churn Performance.png
|   |       ├── Master Sheet.png
|   |
│   └── Telco Data Analysis.pbix  <- Dashboard 
|
├── Data/
│   ├── cleaned_churn_data.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── Deployment/
│   └── deployment.py            <- Streamlit web application
│
├── Models/
│   ├── customer_churn_model.pkl
│   ├── deep_learning_model.keras
│   └── feature_columns.pkl
│
├── PowerBI/
│   ├── Deep Learning/
|   |           ├── Ann_accuracy.csv
|   |           ├── Ann_classification_report.csv
|   |           ├── Ann_predictions.csv
|   |
│   ├── accuracy_metrics.csv
│   ├── classification_report.csv
│   ├── predictions.csv
│   └── training_data.csv
│
├── deep_learning.ipynb          <- ANN pipeline(TensorFlow/Keras)
├── main.ipynb                   <- ML pipeline (LightGBM,GradientBoost,CatBoost,XGBoost..) 
├── README.md
├── requirements.txt
```

