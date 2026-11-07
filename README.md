# Machine Learning Assignment 2 — M.Tech AIML

## a. Problem statement
Build and compare five supervised classification algorithms on one public classification dataset, evaluate them using six metrics, and provide a Streamlit interface that evaluates uploaded test data using saved models.

## b. Dataset description
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), a public binary-classification dataset distributed with scikit-learn and available from the UCI Machine Learning Repository.

- 569 instances and 30 numeric input features
- Target column: `target` (0 = malignant, 1 = benign)
- Stratified 80:20 train/test split with `random_state=42`
- `dataset_full.csv`, `train_data.csv`, and `test_data.csv` are included in this repository.

## c. GitHub Repository Link
https://github.com/2026aj05106-neha/MTech-AIML-ML-Assignment-2

## d. Models used and evaluation

The five classifiers are Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM). StandardScaler is applied through pipelines for Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes use unscaled features.

### Evaluation metrics
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on each model
- **Logistic Regression:** Joint-highest Accuracy, Precision, Recall, F1 and MCC; it also has the highest AUC (0.9954).
- **Decision Tree:** Lowest Accuracy, AUC, F1 and MCC among the five models, although Precision is 0.9565.
- **KNN:** Second-highest F1 (0.9655), with strong Recall (0.9722) and AUC (0.9788).
- **Gaussian Naive Bayes:** AUC is relatively high (0.9878), while Accuracy, F1 and MCC are lower than KNN, Logistic Regression and SVM.
- **SVM:** Ties Logistic Regression on Accuracy, Precision, Recall, F1 and MCC; its AUC (0.9950) is marginally lower.

**Overall winner: Logistic Regression.** Logistic Regression and SVM tie on five metrics. Logistic Regression is selected because its AUC (0.9954) is slightly higher than SVM (0.9950). This conclusion applies to this dataset and this train/test split.

## Streamlit application
The Streamlit application provides:
1. CSV upload for held-out test data.
2. A dropdown to select one of the five saved classifiers.
3. Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix and classification report.
5. Downloadable predictions CSV.

The app loads pre-trained model artifacts and does not retrain on uploaded test data. The uploaded CSV must contain all 30 feature columns and the `target` column; `target_name` is optional.

## Repository structure
```text
.
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── test_data.csv
├── train_data.csv
├── dataset_full.csv
├── model_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Sign in to Streamlit Community Cloud using GitHub.
2. Choose **Create app** / **New app**.
3. Select this repository, the `main` branch, and `app.py`.
4. Deploy, then test by uploading `test_data.csv`.
5. Copy the public app URL into the final assignment PDF after deployment succeeds.

**Live app URL:** To be added after deployment on Streamlit Community Cloud. A live URL cannot be generated before the deployment is completed.

## BITS Virtual Lab execution evidence
Run the project in BITS Virtual Lab and insert the required execution screenshot in the final submission PDF. This screenshot must be captured from the student's own BITS Virtual Lab session.
