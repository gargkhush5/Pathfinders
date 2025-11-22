# Customer Churn Prediction - Machine Learning Project

## Project Overview
This project implements a machine learning solution to predict customer churn using realistic business patterns. The model helps identify customers at risk of discontinuing services, enabling proactive retention strategies.

## Problem Statement
Build a Machine Learning prediction model to:
- Predict customer churn with ~30% realistic churn rate
- Handle imbalanced dataset using appropriate techniques
- Implement comprehensive evaluation metrics
- Provide business insights through confusion matrix and ROC curve analysis

## Dataset Description
**Dataset**: 1000 rows × 7 columns
- **Age**: Customer age (18-70 years)
- **Gender**: Male/Female (categorical)
- **Monthly_UsageHours**: Service usage (5-200 hours)
- **Num_Transactions**: Transaction frequency (1-50)
- **Subscription_Type**: Basic/Premium/Gold (categorical)
- **Complaints**: Number of complaints (0-10)
- **Churn**: Target variable (0 = No, 1 = Yes)

### Realistic Churn Patterns
- **Basic subscribers**: Higher churn (41.9%) - lower engagement
- **Premium/Gold**: Lower churn (29.3%/20.5%) - higher value customers
- **High complaints (8-10)**: 55.4% churn rate
- **Low usage hours (0-50)**: 38% churn rate vs 24.3% for high usage

## Methodology

### Step 1: Data Generation & Preprocessing
```
Data Generation (ds.py)
├── Realistic churn probability modeling
├── Business logic implementation
├── Feature correlation with churn patterns
└── 30.7% churn rate achieved
```

### Step 2: Data Cleaning & Feature Engineering
```
Preprocessing Pipeline
├── Data Loading (customer_churn_data.csv)
├── Missing Value Check
├── Categorical Encoding
│   ├── Gender: Male=0, Female=1
│   └── One-Hot Encoding: Subscription_Type
├── Feature Scaling (StandardScaler)
└── Data Type Conversion
```

### Step 3: Handling Class Imbalance
```
Imbalance Techniques
├── SMOTE (Synthetic Minority Oversampling)
├── Class Weight Balancing
└── Stratified Train-Test Split
```

### Step 4: Model Training & Architecture

#### Data Splitting
- **Training**: 80% (800 samples)
- **Testing**: 20% (200 samples)
- **Stratified split** to maintain class distribution

#### Machine Learning Models
```
Model Pipeline
├── Logistic Regression
│   ├── max_iter=2000
│   ├── class_weight='balanced'
│   └── SMOTE preprocessing
│
└── Random Forest
    ├── n_estimators=200
    ├── max_depth=None
    ├── class_weight='balanced'
    └── Feature importance analysis
```

### Step 5: Model Evaluation

#### Evaluation Metrics
- **Accuracy Score**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve for threshold-independent evaluation

#### Visualization & Analysis
- **Confusion Matrix**: True vs predicted classifications
- **ROC Curve**: True positive rate vs false positive rate
- **Feature Importance**: Random Forest feature ranking
- **Business Impact Analysis**: Churn patterns by customer segments

## File Structure
```


├── customer_churn_data.csv       # Generated dataset
├── HCL_TECH_Path_Finders.ipynb  # Jupyter notebook analysis
└── README.md                     # Project documentation
```

## Key Features
- **Realistic Business Logic**: Churn patterns based on subscription type, complaints, and usage
- **Class Imbalance Handling**: SMOTE oversampling for minority class
- **Comprehensive Evaluation**: Multiple metrics for thorough assessment
- **Business Interpretability**: Clear relationship between features and churn
- **Automated Pipeline**: End-to-end training and evaluation system



                            

