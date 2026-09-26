# Week 4 Task - Supervised Learning
# Iris Flower Classification using Logistic Regression

# ---------------------------------------------------------
# 1. Import Libraries
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# 2. Load the Dataset
# ---------------------------------------------------------

iris = load_iris()

data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

data["target"] = iris.target

print("First five rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# ---------------------------------------------------------
# 3. Check for Missing Values
# ---------------------------------------------------------

print("\nMissing values:")
print(data.isnull().sum())


# ---------------------------------------------------------
# 4. Separate Features and Target
# ---------------------------------------------------------

X = data[iris.feature_names]
y = data["target"]


# ---------------------------------------------------------
# 5. Split Dataset into Training and Testing Data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ---------------------------------------------------------
# 6. Create Machine Learning Model
# ---------------------------------------------------------

# StandardScaler is used to scale the features.
# Logistic Regression is used for classification.

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=200,
        random_state=42
    ))
])


# ---------------------------------------------------------
# 7. Train the Model
# ---------------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# ---------------------------------------------------------
# 8. Make Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)


# ---------------------------------------------------------
# 9. Evaluate the Model
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nModel Evaluation")
print("-------------------------")
print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1 Score :", round(f1, 3))


# ---------------------------------------------------------
# 10. Classification Report
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ---------------------------------------------------------
# 11. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ---------------------------------------------------------
# 12. Cross Validation
# ---------------------------------------------------------

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print(
    "Average Cross Validation Accuracy:",
    round(cv_scores.mean(), 3)
)


# ---------------------------------------------------------
# 13. Display Confusion Matrix
# ---------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(3),
    iris.target_names,
    rotation=20
)

plt.yticks(
    range(3),
    iris.target_names
)

# Add values inside the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# 14. Save Predictions
# ---------------------------------------------------------

results = X_test.copy()

results["Actual"] = [
    iris.target_names[i]
    for i in y_test
]

results["Predicted"] = [
    iris.target_names[i]
    for i in y_pred
]

results.to_csv(
    "iris_classification_predictions.csv",
    index=False
)

print("\nPredictions saved successfully.")


# ---------------------------------------------------------
# 15. Project Completed
# ---------------------------------------------------------

print("\nWeek 4 supervised learning project completed!")
