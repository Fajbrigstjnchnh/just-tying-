# 💰 AI-Powered Loan Approval & Default Risk Prediction System

A Streamlit web app that predicts whether a loan applicant is likely to **default**, using an **XGBoost** model — with every prediction explained using **SHAP**, so the decision isn't a black box.

```
User Inputs
     ↓
Streamlit Website
     ↓
Preprocessing Pipeline (Label Encoding + Scaling)
     ↓
XGBoost Model → Prediction
     ↓
SHAP Explainer → Why Approved / Rejected?
```

---

## 📌 Features

- 11-field applicant input form (age, income, home ownership, loan details, credit history, etc.)
- XGBoost classifier predicting probability of default
- ✅ Approved / ❌ Rejected verdict with confidence %
- SHAP waterfall chart explaining **why** each prediction was made
- Clean, reusable preprocessing pipeline shared between training and inference

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web interface
- **XGBoost** — classification model
- **SHAP** — model explainability
- **Scikit-learn** — preprocessing (LabelEncoder, StandardScaler)
- **Pandas / NumPy** — data handling

---

## 📁 Project Structure

```
loan-default-prediction/
├── data/
│   └── README.md              # Dataset download instructions
├── models/                    # Created after training (model_artifacts.joblib)
├── preprocessing.py           # Shared cleaning + encoding + scaling logic
├── train_model.py             # Trains XGBoost, saves model artifacts
├── app.py                     # Streamlit web app
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

**1. Clone this repository**
```bash
git clone https://github.com/YOUR_USERNAME/loan-default-prediction.git
cd loan-default-prediction
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Get `credit_risk_dataset.csv` from [Kaggle — Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place it inside the `data/` folder.

**5. Train the model**
```bash
python train_model.py
```
This prints accuracy/ROC-AUC and saves `models/model_artifacts.joblib`.

**6. Launch the app**
```bash
streamlit run app.py
```
Open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## ☁️ Deploy for Free (Streamlit Community Cloud)

1. Push this whole project to a **public GitHub repo** — make sure `data/credit_risk_dataset.csv` and `models/model_artifacts.joblib` are included (run `train_model.py` locally first, then commit the generated `models/` folder).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **New app**, select your repo, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs everything from `requirements.txt` automatically.
5. You'll get a public link like `https://your-app-name.streamlit.app` — put this in your resume/portfolio.

---

## 🧠 How It Works

| Step | Detail |
|---|---|
| **Preprocessing** | Categorical fields (home ownership, loan intent, loan grade, previous default) are Label Encoded. Numeric fields are scaled with `StandardScaler`. The exact same encoders/scaler from training are reused at prediction time. |
| **Model** | XGBoost classifier (`n_estimators=300`, `max_depth=5`, `learning_rate=0.05`) trained on an 80/20 train-test split. |
| **Explainability** | `shap.TreeExplainer` computes each feature's contribution to that specific prediction, visualized as a waterfall chart — red pushes toward default, blue pushes toward approval. |

---

## 💡 What This Project Demonstrates

- End-to-end ML pipeline: raw data → cleaning → feature engineering → model → deployment
- Production-style code structure (shared preprocessing module, not copy-pasted logic)
- Model explainability with SHAP — increasingly expected in finance/credit ML roles
- Deploying a live, usable ML web app, not just a notebook
