"""
train_model.py
Trains the XGBoost loan-default model on the Kaggle "Credit Risk Dataset"
and saves everything the Streamlit app needs (model + encoders + scaler)
into a single file: models/model_artifacts.joblib

Run this ONCE after placing credit_risk_dataset.csv inside the data/ folder:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

from preprocessing import (
    clean_data,
    fit_encoders,
    fit_scaler,
    FEATURE_ORDER,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
)

DATA_PATH = "data/credit_risk_dataset.csv"
TARGET_COL = "loan_status"          # 1 = defaulted, 0 = did not default
ARTIFACT_PATH = "models/model_artifacts.joblib"


def main():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {df.shape}")

    df = clean_data(df)
    print(f"Cleaned shape: {df.shape}")

    df, encoders = fit_encoders(df)
    df, scaler = fit_scaler(df)

    X = df[FEATURE_ORDER]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Model Evaluation on Test Set ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "encoders": encoders,
            "scaler": scaler,
            "feature_order": FEATURE_ORDER,
            "categorical_cols": CATEGORICAL_COLS,
            "numeric_cols": NUMERIC_COLS,
        },
        ARTIFACT_PATH,
    )
    print(f"\nSaved model + preprocessing artifacts to: {ARTIFACT_PATH}")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
