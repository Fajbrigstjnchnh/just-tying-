"""
app.py
AI-Powered Loan Approval & Default Risk Prediction System
Streamlit website: takes applicant details, predicts default risk with
XGBoost, and explains the decision with SHAP.

Run with:
    streamlit run app.py
"""

import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

from preprocessing import transform_input

ARTIFACT_PATH = "models/model_artifacts.joblib"

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💰",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACT_PATH)


try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Please run `python train_model.py` "
        "first to train the model (after placing the dataset in data/)."
    )
    st.stop()

model = artifacts["model"]
encoders = artifacts["encoders"]
scaler = artifacts["scaler"]

st.title("💰 AI-Powered Loan Approval & Default Risk Prediction")
st.write(
    "Enter applicant details below. The model predicts default risk and "
    "explains **why** using SHAP."
)

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input(
            "Annual Income ($)", min_value=0, value=50000, step=1000
        )
        home_ownership = st.selectbox(
            "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )
        emp_length = st.number_input(
            "Employment Length (years)", min_value=0.0, max_value=60.0, value=5.0
        )
        loan_intent = st.selectbox(
            "Loan Intent",
            [
                "PERSONAL",
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION",
            ],
        )
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

    with col2:
        loan_amnt = st.number_input(
            "Loan Amount ($)", min_value=0, value=10000, step=500
        )
        loan_int_rate = st.number_input(
            "Interest Rate (%)", min_value=0.0, max_value=40.0, value=11.5
        )
        loan_percent_income = st.number_input(
            "Loan Percent Income (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            help="Loan amount as a fraction of annual income, e.g. 0.20 = 20%",
        )
        default_on_file = st.selectbox("Previous Default", ["Y", "N"])
        cred_hist_length = st.number_input(
            "Credit History Length (years)", min_value=0, max_value=50, value=5
        )

    submitted = st.form_submit_button("🔵 Predict")

if submitted:
    raw_input = {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home_ownership,
        "person_emp_length": emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": default_on_file,
        "cb_person_cred_hist_length": cred_hist_length,
    }

    X_input = transform_input(raw_input, encoders, scaler)

    proba_default = model.predict_proba(X_input)[0][1]
    prediction = model.predict(X_input)[0]

    st.divider()

    if prediction == 1:
        st.error(
            f"❌ Loan Rejected — High Default Risk "
            f"({proba_default * 100:.1f}% predicted probability of default)"
        )
    else:
        st.success(
            f"✅ Loan Approved — Low Default Risk "
            f"({proba_default * 100:.1f}% predicted probability of default)"
        )

    st.subheader("🔍 Why this decision?")
    st.caption(
        "Bars pushing the prediction higher (toward default) are shown in red. "
        "Bars pushing it lower (toward approval) are shown in blue."
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "Model: XGBoost · Explainability: SHAP · Dataset: Kaggle Credit Risk Dataset"
)
