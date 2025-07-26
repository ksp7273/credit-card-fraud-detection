# Comprehensive Guide to Fraud Detection Project

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

## Modified `fraud_detection.py`
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib

# 1. Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, scaler

# 2. Handle class imbalance with SMOTE
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    return X_resampled, y_resampled

# 3. Train LightGBM model with hyperparameter tuning
def train_model(X_train, y_train):
    param_grid = {
        'num_leaves': [31],
        'learning_rate': [0.05],
        'n_estimators': [100],
        'min_child_samples': [20]
    }
    lgb_clf = lgb.LGBMClassifier(random_state=42, force_col_wise=True, verbose=-1)
    grid_search = GridSearchCV(
        estimator=lgb_clf,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# 4. Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    return precision, recall

# 5. Save model
def save_model(model, scaler, model_path='fraud_model.pkl', scaler_path='scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

# Main execution
def main():
    file_path = 'creditcard.csv'
    X, y, scaler = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    print("Training LightGBM model...")
    model, best_params = train_model(X_train_resampled, y_train_resampled)
    print("Best parameters:", best_params)
    print("\nEvaluating model...")
    precision, recall = evaluate_model(model, X_test, y_test)
    save_model(model, scaler)
    results_df = pd.DataFrame({'precision': precision, 'recall': recall})
    results_df.to_csv('precision_recall_results.csv', index=False)
    print("Precision-recall data saved for Tableau visualization")

if __name__ == "__main__":
    main()
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
   ```python
   import json
   import joblib
   import pandas as pd
   import numpy as np

   def model_fn(model_dir):
       model = joblib.load(f"{model_dir}/fraud_model.pkl")
       scaler = joblib.load(f"{model_dir}/scaler.pkl")
       return {"model": model, "scaler": scaler}

   def input_fn(request_body, request_content_type):
       if request_content_type == "application/json":
           data = json.loads(request_body)
           df = pd.DataFrame(data)
           return df
       raise ValueError("Unsupported content type")

   def predict_fn(input_data, model_dict):
       model = model_dict["model"]
       scaler = model_dict["scaler"]
       X_scaled = scaler.transform(input_data)
       predictions = model.predict_proba(X_scaled)[:, 1]
       return predictions

   def output_fn(prediction, accept):
       if accept == "application/json":
           return json.dumps(prediction.tolist())
       raise ValueError("Unsupported accept type")
   ```
4. **Deploy Script**:
   ```python
   import sagemaker
   from sagemaker.sklearn.model import SKLearnModel
   from sagemaker import get_execution_role
   import boto3

   session = sagemaker.Session()
   role = get_execution_role()
   model = SKLearnModel(
       model_data="s3://credit-fraud-model/model.tar.gz",
       role=role,
       entry_point="inference.py",
       framework_version="1.2-1",
       py_version="py3"
   )
   predictor = model.deploy(
       instance_type="ml.t2.medium",
       initial_instance_count=1,
       endpoint_name="fraud-detection-endpoint"
   )
   print(f"Endpoint deployed: {predictor.endpoint_name}")
   ```
5. **Test Endpoint**:
   ```python
   import boto3
   import json
   import pandas as pd

   client = boto3.client("sagemaker-runtime", region_name="us-east-1")
   test_data = pd.read_csv("creditcard.csv").drop(["Class"], axis=1).head(1).to_dict(orient="records")
   response = client.invoke_endpoint(
       EndpointName="fraud-detection-endpoint",
       ContentType="application/json",
       Accept="application/json",
       Body=json.dumps(test_data)
   )
   predictions = json.loads(response["Body"].read().decode())
   print(f"Predictions: {predictions}")
   ```

## Visualizing in Tableau
1. Import `precision_recall_results.csv`.
2. Create precision-recall curve (Recall on Columns, Precision on Rows, Line chart).
3. Visualize classification report metrics.
4. Save as `fraud_detection.twb`.