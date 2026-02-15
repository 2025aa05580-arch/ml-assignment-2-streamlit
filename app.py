import os
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("ML Assignment 2 — Classification Models Demo")

MODEL_DIR = "model/saved_models"
TARGET_COL = "label"

# --- Sidebar ---
st.sidebar.header("Controls")
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}
selected_model_name = st.sidebar.selectbox("Select Model", list(model_map.keys()))
selected_model_file = model_map[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Upload a CSV that includes a 'label' column.")

# --- Show stored comparison metrics if available ---
metrics_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
st.subheader("Model Comparison Table (from your evaluation script)")

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning("metrics_summary.csv not found. Run: python model/evaluate_models.py")

st.markdown("---")

# --- Upload ---
st.subheader("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV with features + label column", type=["csv"])

def compute_metrics(y_true, y_pred, y_prob=None):
    out = {}
    out["Accuracy"] = accuracy_score(y_true, y_pred)
    out["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)
    out["MCC"] = matthews_corrcoef(y_true, y_pred)

    # AUC only if probabilities available
    if y_prob is not None:
        out["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        out["AUC"] = None
    return out

if uploaded_file is None:
    st.info("Upload a CSV to run predictions.")
    st.stop()

# --- Read uploaded dataset ---
df = pd.read_csv(uploaded_file)

if TARGET_COL not in df.columns:
    st.error(f"Uploaded CSV must contain target column '{TARGET_COL}'.")
    st.stop()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# --- Load model ---
model_path = os.path.join(MODEL_DIR, selected_model_file)
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Train models first.")
    st.stop()

model = joblib.load(model_path)

# --- Predict ---
y_pred = model.predict(X)

# Probability for AUC
y_prob = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        y_prob = proba[:, 1]

# --- Metrics ---
st.subheader(f"Metrics for: {selected_model_name}")
metrics = compute_metrics(y, y_pred, y_prob)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
c2.metric("Precision", f"{metrics['Precision']:.4f}")
c3.metric("Recall", f"{metrics['Recall']:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("F1", f"{metrics['F1']:.4f}")
c5.metric("MCC", f"{metrics['MCC']:.4f}")
c6.metric("AUC", "N/A" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}")

st.markdown("---")

# --- Report + Confusion Matrix ---
st.subheader("Classification Report")
st.text(classification_report(y, y_pred, zero_division=0))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
st.write(cm)

st.success("✅ Done! Try switching models from the sidebar.")
