import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation", page_icon="🧪", layout="wide")
st.title("Machine Learning Classification Model Evaluator")
st.write("Upload the held-out test CSV, select a trained classifier, and evaluate its predictions.")

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"])
selected = st.selectbox("Select classification model", list(MODEL_FILES.keys()))

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.subheader("Uploaded data preview")
    st.dataframe(data.head(), use_container_width=True)

    if "target" not in data.columns:
        st.error("The uploaded CSV must contain the target column named 'target'.")
        st.stop()

    target = data["target"]
    feature_data = data.drop(columns=[c for c in ["target", "target_name"] if c in data.columns])

    model = joblib.load(MODEL_DIR / MODEL_FILES[selected])
    expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None

    if expected_features is not None and list(feature_data.columns) != expected_features:
        missing = sorted(set(expected_features) - set(feature_data.columns))
        extra = sorted(set(feature_data.columns) - set(expected_features))
        st.error(f"Feature columns do not match the training data. Missing: {missing}; Extra: {extra}.")
        st.stop()

    predictions = model.predict(feature_data)
    probabilities = model.predict_proba(feature_data)[:, 1]

    st.subheader("Evaluation metrics")
    try:
        auc = roc_auc_score(target, probabilities)
    except ValueError:
        auc = float("nan")
    values = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": auc,
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1 Score": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }
    cols = st.columns(3)
    for col, (name, value) in zip(cols * 2, values.items()):
        col.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")

    st.subheader("Confusion matrix")
    cm = confusion_matrix(target, predictions)
    st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

    st.subheader("Classification report")
    report = classification_report(target, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    output = data.copy()
    output["predicted_target"] = predictions
    st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")
else:
    st.info("Upload test_data.csv from the project folder to view evaluation results.")
    st.caption("The app loads saved models only. It does not retrain models on uploaded test data.")
