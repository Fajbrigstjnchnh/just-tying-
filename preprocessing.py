"""
preprocessing.py
Shared preprocessing logic used by both train_model.py and app.py.
Keeping this in one place guarantees the app preprocesses user input
EXACTLY the same way the model was trained on.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Categorical columns -> Label Encoded
CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

# Numeric columns -> Scaled
NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

# Final column order fed into the model (must always match training order)
FEATURE_ORDER = NUMERIC_COLS + CATEGORICAL_COLS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning applied only during training (raw CSV is messy)."""
    df = df.copy()

    # Drop a few known data-entry errors in this dataset (e.g. age 144)
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]

    # Fill missing numeric values with the median
    for col in ["person_emp_length", "loan_int_rate"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.reset_index(drop=True)
    return df


def fit_encoders(df: pd.DataFrame):
    """Fit a LabelEncoder per categorical column. Returns transformed df + encoders dict."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def fit_scaler(df: pd.DataFrame):
    """Fit a StandardScaler on numeric columns. Returns transformed df + scaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    return df, scaler


def transform_input(raw_dict: dict, encoders: dict, scaler: StandardScaler) -> pd.DataFrame:
    """
    Takes ONE raw user input (from the Streamlit form) as a dict with human-readable
    values, applies the SAME encoders + scaler fitted during training, and returns
    a single-row DataFrame ready for model.predict().
    """
    df = pd.DataFrame([raw_dict])

    for col in CATEGORICAL_COLS:
        le = encoders[col]
        value = str(df.at[0, col])
        if value not in le.classes_:
            # Unseen category at inference time -> fall back to the most common
            # known class instead of crashing the app.
            value = le.classes_[0]
        df[col] = le.transform([value])

    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    return df[FEATURE_ORDER]
