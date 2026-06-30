# Credit Card Fraud Detection

End-to-end ML pipeline for detecting fraudulent credit card transactions on a highly imbalanced dataset (fraud = 0.17% of transactions). Compares classical ML and deep learning approaches, and benchmarks two different strategies for handling class imbalance.

## Results

| Model              | AUC-PR | AUC-ROC |
|---------------------|--------|---------|
| XGBoost (class_weight) | 0.888  | 0.971   |
| XGBoost (SMOTE)        | 0.872  | 0.981   |
| RF (class_weight)      | 0.826  | 0.981   |
| RF (SMOTE)              | 0.816  | 0.980   |
| MLP (SMOTE)             | 0.813  | 0.970   |
| LR (SMOTE)              | 0.725  | 0.970   |
| LR (class_weight)       | 0.719  | 0.972   |

AUC-PR (Precision-Recall) is used as the primary metric rather than accuracy, since accuracy is misleading on a dataset that's 99.8% one class — a model predicting "no fraud" every time would score 99.8% accuracy while catching zero fraud.

## What this project covers

- Exploratory data analysis on transaction amounts and class distribution
- Two imbalance-handling strategies compared directly: SMOTE oversampling vs. cost-sensitive class weighting
- Four models: Logistic Regression, Random Forest, XGBoost, and a Keras MLP
- Evaluation via AUC-PR, AUC-ROC, and full classification reports
- SHAP-based interpretability to identify which features drive fraud predictions

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 transactions made by European cardholders, with PCA-transformed features (V1–V28) plus `Time` and `Amount`. 492 transactions (0.172%) are fraudulent.

## How to run

### Option 1: Google Colab (recommended)

1. Open the notebook in Colab: `[link to your .ipynb once uploaded]`
2. Get a Kaggle API token: go to [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New Token" → this downloads `kaggle.json`
3. Run the first cell and upload `kaggle.json` when prompted
4. Run all remaining cells in order

### Option 2: Local setup

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

Place your `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows), then:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
jupyter notebook fraud_detection.ipynb
```
