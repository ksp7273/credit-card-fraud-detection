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
        'num_leaves': [31],  # Reduced for testing
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
    file_path = 'creditcard.csv'  # Verify this path
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