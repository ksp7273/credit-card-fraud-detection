# Steps to deploy Credit Card Fraud Detection Project

## Introduction
This guide details downloading the Kaggle Credit Card Fraud Detection dataset, setting up GitHub Codespace, running the fraud detection project with LightGBM, deploying on AWS SageMaker, and visualizing insights in Tableau. It includes code files: `fraud_detection.py`, `inference.py`, `deploy_sagemaker.py`, and `test_endpoint.py`.

## Getting Data from Kaggle
1. **Create Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com).
2. **Generate API Token**: Settings > API > Create New Token to download `kaggle.json`.
3. **Install Kaggle CLI**:
   ```bash
   pip install kaggle
   ```
4. **Download Dataset**:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud
   unzip creditcardfraud.zip
   ```

## Adding `kaggle.json` to GitHub Codespace
1. **Upload `kaggle.json`**:
   - Drag and drop into file explorer.
   - Or create manually: Copy contents, create `kaggle.json`, paste, and save.
2. **Move to `~/.kaggle/`**:
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. **Secure File**:
   ```bash
   echo ".kaggle/" >> .gitignore
   echo "kaggle.json" >> .gitignore
   ```

## Running the Fraud Detection Project
1. **Set Up Repository**: Create a GitHub repository and start a Codespace.
2. **Add Script**: Use `fraud_detection.py` (below).
3. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn lightgbm imblearn joblib
   ```
4. **Run Script**:
   ```bash
   python fraud_detection.py
   ```



**How It Works**:
1. **Imports**: Libraries for data, modeling, and serialization.
2. **load_and_preprocess_data**: Loads and scales `creditcard.csv`.
3. **apply_smote**: Balances classes using SMOTE.
4. **train_model**: Trains LightGBM with optimized settings.
5. **evaluate_model**: Computes metrics and precision-recall data.
6. **save_model**: Saves model and scaler.

## LightGBM Model
**Why Chosen**: Efficient, scalable, handles imbalanced data, SageMaker-compatible.

**Architecture**:
- **Histogram-Based Binning**: Discretizes features (7,650 bins).
- **Leaf-Wise Growth**: Splits leaves for maximum loss reduction.
- **GOSS**: Prioritizes high-gradient samples.
- **EFB**: Groups correlated features.

**Advantages**:
- Fast and scalable.
- High accuracy.
- Flexible tuning.

**Disadvantages**:
- Overfitting risk.
- Memory-intensive with SMOTE.
- Less interpretable.

## Deploying on AWS SageMaker
1. **Prepare Files**:
   ```bash
   tar -czf model.tar.gz fraud_model.pkl scaler.pkl inference.py
   ```
2. **Upload to S3**:
   ```bash
   aws s3 mb s3://credit-fraud-model
   aws s3 cp model.tar.gz s3://credit-fraud-model/model.tar.gz
   ```
3. **Inference Script**:
    Run  Inference Script
4. **Deploy Script**:
   Run Deploy Script
5. **Test Endpoint**:
   Run Test Endpoint

## Visualizing in Tableau
1. Import `precision_recall_results.csv`.
2. Create precision-recall curve (Recall on Columns, Precision on Rows, Line chart).
3. Visualize classification report metrics.
4. Save as `fraud_detection.twb`.
