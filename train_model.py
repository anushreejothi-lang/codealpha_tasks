# ============================================
# CodeAlpha - Machine Learning Task 1
# Credit Scoring Model
# ============================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================
# 1. LOAD DATASET
# ============================================

print("Loading UCI Credit Approval dataset...")

data = pd.read_csv(
    "dataset/crx.data",
    header=None,
    na_values="?"
)

print("Dataset loaded successfully!")
print("Dataset Shape:", data.shape)


# ============================================
# 2. GIVE COLUMN NAMES
# ============================================

columns = [
    "A1", "A2", "A3", "A4", "A5",
    "A6", "A7", "A8", "A9", "A10",
    "A11", "A12", "A13", "A14", "A15", "A16"
]

data.columns = columns

print("\nFirst 5 rows:")
print(data.head())


# ============================================
# 3. CLEAN TARGET COLUMN
# ============================================

data["A16"] = data["A16"].astype(str).str.strip()

# + = Approved / Creditworthy
# - = Rejected / Not Creditworthy

data["creditworthy"] = data["A16"].map({
    "+": 1,
    "-": 0
})

# Remove rows where target is missing or invalid
data = data.dropna(subset=["creditworthy"])


# ============================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================

X = data.drop(["A16", "creditworthy"], axis=1)
y = data["creditworthy"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================
# 5. FIND NUMERICAL AND CATEGORICAL COLUMNS
# ============================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================
# 6. PREPROCESS NUMERICAL DATA
# ============================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================
# 7. PREPROCESS CATEGORICAL DATA
# ============================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ============================================
# 8. COMBINE PREPROCESSING
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ============================================
# 9. CREATE LOGISTIC REGRESSION MODEL
# ============================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ]
)


# ============================================
# 10. SPLIT DATA INTO TRAINING AND TESTING
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ============================================
# 11. TRAIN MODEL
# ============================================

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================
# 12. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# Probability needed for ROC-AUC
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================
# 13. CALCULATE EVALUATION METRICS
# ============================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================
# 14. DISPLAY RESULTS
# ============================================

print("\n============================================")
print("        CREDIT SCORING MODEL RESULTS")
print("============================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ============================================
# 15. CLASSIFICATION REPORT
# ============================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Creditworthy",
            "Creditworthy"
        ],
        zero_division=0
    )
)


# ============================================
# 16. CONFUSION MATRIX
# ============================================

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================
# 17. SAVE TRAINED MODEL
# ============================================

joblib.dump(
    model,
    "credit_scoring_model.pkl"
)

print("\n============================================")
print("Model saved successfully!")
print("File: credit_scoring_model.pkl")
print("============================================")