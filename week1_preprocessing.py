# ============================================================
# WEEK 1: DATA ACQUISITION, CLEANING AND PREPROCESSING
# Dataset: UCI Adult / Census Income Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. DATA ACQUISITION
# ------------------------------------------------------------

# Column names according to the UCI Adult dataset
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

# Official UCI dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# Load dataset
df = pd.read_csv(
    url,
    names=columns,
    skipinitialspace=True,
    na_values="?"
)

# Keep a copy of the original dataset
raw_df = df.copy()

print("=" * 60)
print("DATASET SUCCESSFULLY LOADED")
print("=" * 60)

print("Dataset shape:", df.shape)
print()


# ------------------------------------------------------------
# 2. INITIAL DATA EXPLORATION
# ------------------------------------------------------------

print("=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(df.head())
print()


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()
print()


print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe(include="all").T)
print()


# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df) * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing Percentage": missing_percentage
})

print(missing_report[missing_report["Missing Count"] > 0])
print()


# ------------------------------------------------------------
# 4. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

print("=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)
print()


# ------------------------------------------------------------
# 5. CLEAN WHITESPACE FROM CATEGORICAL VARIABLES
# ------------------------------------------------------------

print("=" * 60)
print("CLEANING CATEGORICAL VARIABLES")
print("=" * 60)

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns

for column in categorical_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

print("Whitespace normalization completed.")
print()


# ------------------------------------------------------------
# 6. HANDLE MISSING CATEGORICAL VALUES
# ------------------------------------------------------------

print("=" * 60)
print("HANDLING MISSING CATEGORICAL VALUES")
print("=" * 60)

missing_categorical_columns = [
    "workclass",
    "occupation",
    "native_country"
]

for column in missing_categorical_columns:
    df[column] = df[column].fillna("Unknown")

print("Missing categorical values replaced with 'Unknown'.")
print()


# ------------------------------------------------------------
# 7. NORMALIZE THE TARGET VARIABLE
# ------------------------------------------------------------

print("=" * 60)
print("TARGET VARIABLE CLEANING")
print("=" * 60)

# Remove unnecessary period from income labels if present
df["income"] = (
    df["income"]
    .astype("string")
    .str.replace(".", "", regex=False)
    .str.strip()
)

print("Income categories:")
print(df["income"].value_counts())
print()


# Check for unexpected income categories
valid_income_categories = {
    "<=50K",
    ">50K"
}

unexpected_categories = set(
    df["income"].dropna().unique()
) - valid_income_categories

print("Unexpected income categories:")
print(unexpected_categories)
print()


# ------------------------------------------------------------
# 8. NUMERIC DATA VALIDATION
# ------------------------------------------------------------

print("=" * 60)
print("NUMERIC RANGE CHECKS")
print("=" * 60)

print(
    "Age values <= 0:",
    (df["age"] <= 0).sum()
)

print(
    "Education numbers outside 1-16:",
    (~df["education_num"].between(1, 16)).sum()
)

print(
    "Negative capital gain values:",
    (df["capital_gain"] < 0).sum()
)

print(
    "Negative capital loss values:",
    (df["capital_loss"] < 0).sum()
)

print(
    "Negative working hours:",
    (df["hours_per_week"] < 0).sum()
)

print()


# ------------------------------------------------------------
# 9. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("=" * 60)
print("OUTLIER ANALYSIS USING IQR")
print("=" * 60)

numeric_columns = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week"
]

outlier_results = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    outlier_count = outlier_mask.sum()
    outlier_percentage = (
        outlier_count / len(df) * 100
    )

    outlier_results.append({
        "Variable": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count,
        "Outlier Percentage": round(
            outlier_percentage, 2
        )
    })

outlier_report = pd.DataFrame(outlier_results)

print(outlier_report)
print()


# ------------------------------------------------------------
# 10. VISUALIZE OUTLIERS
# ------------------------------------------------------------

print("=" * 60)
print("GENERATING OUTLIER VISUALIZATIONS")
print("=" * 60)

for column in numeric_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title(
        f"Boxplot of {column}"
    )

    plt.xlabel(column)

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 11. HISTOGRAMS FOR NUMERIC VARIABLES
# ------------------------------------------------------------

for column in numeric_columns:

    plt.figure(figsize=(8, 4))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 12. FINAL DATA QUALITY CHECK
# ------------------------------------------------------------

print("=" * 60)
print("FINAL DATA QUALITY CHECK")
print("=" * 60)

print("Final dataset shape:", df.shape)

print(
    "Remaining missing cells:",
    df.isnull().sum().sum()
)

print(
    "Remaining duplicate rows:",
    df.duplicated().sum()
)

print()

print("Missing values by column:")
print(df.isnull().sum())
print()


# ------------------------------------------------------------
# 13. BEFORE VS AFTER COMPARISON
# ------------------------------------------------------------

print("=" * 60)
print("BEFORE VS AFTER CLEANING")
print("=" * 60)

comparison = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Missing Cells",
        "Duplicate Rows"
    ],

    "Before Cleaning": [
        raw_df.shape[0],
        raw_df.shape[1],
        raw_df.isnull().sum().sum(),
        raw_df.duplicated().sum()
    ],

    "After Cleaning": [
        df.shape[0],
        df.shape[1],
        df.isnull().sum().sum(),
        df.duplicated().sum()
    ]
})

print(comparison)
print()


# ------------------------------------------------------------
# 14. SAVE CLEANED DATASET
# ------------------------------------------------------------

output_file = "adult_cleaned.csv"

df.to_csv(
    output_file,
    index=False
)

print("=" * 60)
print("CLEANED DATASET SAVED")
print("=" * 60)

print(
    f"Cleaned dataset saved as: {output_file}"
)

print()
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
