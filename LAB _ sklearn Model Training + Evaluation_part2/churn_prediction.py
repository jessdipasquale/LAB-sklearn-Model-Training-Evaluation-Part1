# churn_prediction.py
# Part 2: Unguided Exercise - Customer Churn Prediction (Telco Dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
# Step 1: Load + Explore

DATA_PATH = "/Users/user/Desktop/Ironhack/Week1/20260204/LAB _ sklearn Model Training + Evaluation_part2/Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n=== BASIC INFO ===")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES (raw) ===")
print(df.isna().sum().sort_values(ascending=False).head(15))

print("\n=== TARGET DISTRIBUTION ===")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))

# Quick visuals 
plt.figure()
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Counts")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

if "tenure" in df.columns:
    plt.figure()
    df.boxplot(column="tenure", by="Churn")
    plt.title("Tenure by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (months)")
    plt.tight_layout()
    plt.show()

if "MonthlyCharges" in df.columns:
    plt.figure()
    df.boxplot(column="MonthlyCharges", by="Churn")
    plt.title("MonthlyCharges by Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.show()


# Step 2: Data Preprocessing

# Cleaning for this dataset:
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# Drop customerID (not useful for prediction)
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

print("\n=== MISSING VALUES (after TotalCharges conversion + drop customerID) ===")
print(df.isna().sum().sort_values(ascending=False).head(15))

# - For numeric columns: fill with median
# - For categorical columns: fill with mode
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

# Remove target from categorical list if present
if "Churn" in categorical_cols:
    categorical_cols.remove("Churn")

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Separate X and y
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})  # convert target to 0/1

print("\n=== CHECK y VALUES ===")
print(y.value_counts())

# Build preprocessing:
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
)

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("\n=== SPLIT SIZES ===")
print("Train:", X_train.shape, "Test:", X_test.shape)

# Step 4-6: Train KNN + Evaluate + Try K values
k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

for k in k_values:
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    results.append({"k": k, "accuracy": acc, "precision": prec, "recall": rec})

results_df = pd.DataFrame(results).sort_values(by="recall", ascending=False)

print("\n=== RESULTS (sorted by RECALL, higher is better for catching churners) ===")
print(results_df.to_string(index=False))

best_row = results_df.iloc[0]
best_k = int(best_row["k"])

print(f"\nBest K by recall: {best_k}")
print("Best metrics:", best_row.to_dict())

# Final evaluation with best K
best_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ]
)

best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

print("\n=== EVALUATION WITH BEST K ===")
print("Accuracy :", accuracy_score(y_test, best_pred))
print("Precision:", precision_score(y_test, best_pred, zero_division=0))
print("Recall   :", recall_score(y_test, best_pred, zero_division=0))

cm = confusion_matrix(y_test, best_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, best_pred, target_names=["No Churn", "Churn"]))

# Confusion matrix plot (simple)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (Best K)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, int(val), ha="center", va="center")
plt.tight_layout()
plt.show()

# =========================
# Step 7: Basic feature insights (simple exploration)
# =========================
# KNN doesn't have "feature importance" like trees.
# But we can explore distributions of key features vs Churn.

def churn_rate_by_category(data, col):
    table = data.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
    return table

print("\n=== SIMPLE FEATURE INSIGHTS ===")
for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
    if col in df.columns:
        rates = churn_rate_by_category(df, col)
        print(f"\nChurn rate by {col}:")
        print(rates)

print("\n=== DONE ===")

"""
ANALYSIS AND CONCLUSIONS

Model Performance:
The KNN model was trained to predict customer churn.
Recall was used as the main metric to identify customers likely to churn.

Findings:
Customers with short tenure and month-to-month contracts churn more.
High monthly charges and lack of technical support are also related to churn.

Business Recommendations:
Offer discounts for long-term contracts.
Provide technical support to customers at risk.

Limitations:
KNN does not provide feature importance.
Results depend on data preprocessing and scaling.
"""