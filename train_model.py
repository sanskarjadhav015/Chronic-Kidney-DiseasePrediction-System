import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_save_model():
    print("=== Starting Chronic Kidney Disease Model Retraining Pipeline ===")
    
    # 1. Load Dataset
    data_path = "kidney_disease.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Select Relevant Features
    important_columns = ['age', 'bp', 'sg', 'al', 'hemo', 'sc', 'htn', 'dm', 'cad', 'appet', 'pc', 'classification']
    df = df[important_columns]
    
    # 3. Clean String Whitespaces
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.replace(r'\t', '', regex=True)
        
    # 4. Impute Missing Values
    # Numerical → Median
    df['age'] = df['age'].fillna(df['age'].median())
    df['bp'] = df['bp'].fillna(df['bp'].median())
    df['hemo'] = df['hemo'].fillna(df['hemo'].median())
    df['sc'] = df['sc'].fillna(df['sc'].median())
    
    # Discrete / Categorical → Mode
    df['sg'] = df['sg'].fillna(df['sg'].mode()[0])
    df['al'] = df['al'].fillna(df['al'].mode()[0])
    df['htn'] = df['htn'].fillna(df['htn'].mode()[0])
    df['dm'] = df['dm'].fillna(df['dm'].mode()[0])
    df['cad'] = df['cad'].fillna(df['cad'].mode()[0])
    df['appet'] = df['appet'].fillna(df['appet'].mode()[0])
    df['pc'] = df['pc'].fillna(df['pc'].mode()[0])
    
    # 5. Map Categorical Values to Binary
    df['htn'] = df['htn'].map({'yes': 1, 'no': 0})
    df['dm'] = df['dm'].map({'yes': 1, 'no': 0})
    df['cad'] = df['cad'].map({'yes': 1, 'no': 0})
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    
    # 6. Separate Features and Target
    X = df.drop('classification', axis=1)
    y = df['classification']
    
    # 7. Leak-Free Train-Test Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Initial Split -> Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 8. Fit MinMaxScaler on X_train ONLY, transform X_train and X_test
    numeric_cols = ['age', 'bp', 'sg', 'al', 'hemo', 'sc']
    scaler = MinMaxScaler()
    
    # Create copies to avoid pandas warnings
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 9. Apply SMOTE on X_train ONLY
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE -> Resampled Train shape: {X_train_res.shape}")
    
    # 10. Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)
    
    # 11. Evaluate on Unseen Test Set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Model Evaluation on Test Set:")
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # 12. Save Scaler and Model Artifacts
    os.makedirs("models", exist_ok=True)
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(model, open("models/model_gbc.pkl", "wb"))
    print("\n[+] Retraining complete! Saved 'models/scaler.pkl' and 'models/model_gbc.pkl'.")

if __name__ == "__main__":
    train_and_save_model()
