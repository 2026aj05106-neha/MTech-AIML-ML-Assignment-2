"""Train the five required classifiers and save metrics and fitted models."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).parent
data = pd.read_csv(ROOT / "dataset_full.csv")
X = data.drop(columns=["target", "target_name"])
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True, kernel="rbf", random_state=42))]),
}
(ROOT / "model").mkdir(exist_ok=True)
results = []
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    score = estimator.predict_proba(X_test)[:, 1]
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, score),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    filename = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Support Vector Machine": "svm",
    }[name]
    joblib.dump(estimator, ROOT / "model" / f"{filename}.joblib")

pd.DataFrame(results).to_csv(ROOT / "model_metrics.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))
