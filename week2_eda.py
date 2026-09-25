# ============================================================
# WEEK 2 TASK
# EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# Dataset: UCI Adult / Census Income Dataset
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

pd.set_option("display.max_columns", None)

print("=" * 70)
print("WEEK 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

# If you have the cleaned Week 1 CSV file, keep it in the
# same folder as this Python file.

file_path = "adult_cleaned.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ------------------------------------------------------------
# 3. INITIAL DATA EXPLORATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST FIVE ROWS")
print("=" * 70)

print(df.head())


print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(df.info())


print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERICAL DESCRIPTIVE STATISTICS")
print("=" * 70)

numeric_summary = df.describe()

print(numeric_summary)


print("\n" + "=" * 70)
print("CATEGORICAL DESCRIPTIVE STATISTICS")
print("=" * 70)

categorical_summary = df.describe(include="object")

print(categorical_summary)


# ------------------------------------------------------------
# 5. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df) * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing_values,
    "Missing Percentage": missing_percentage
})

print(missing_report)


# ------------------------------------------------------------
# 6. CHECK DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ------------------------------------------------------------
# 7. INCOME DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INCOME DISTRIBUTION")
print("=" * 70)

income_counts = df["income"].value_counts()

print(income_counts)

income_percentages = (
    df["income"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nIncome percentages:")
print(income_percentages)


# ------------------------------------------------------------
# VISUALIZATION 1
# Income Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="income"
)

plt.title(
    "Distribution of Income Classes",
    fontsize=15
)

plt.xlabel("Income Class")
plt.ylabel("Number of Individuals")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    "01_income_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 8. AGE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AGE ANALYSIS")
print("=" * 70)

print(df["age"].describe())


# ------------------------------------------------------------
# VISUALIZATION 2
# Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title(
    "Distribution of Age",
    fontsize=15
)

plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "02_age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# VISUALIZATION 3
# Age vs Income
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="income",
    y="age"
)

plt.title(
    "Age Distribution by Income Class",
    fontsize=15
)

plt.xlabel("Income Class")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    "03_age_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 9. EDUCATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDUCATION ANALYSIS")
print("=" * 70)

education_counts = (
    df["education"]
    .value_counts()
)

print(education_counts)


# ------------------------------------------------------------
# VISUALIZATION 4
# Education Distribution
# ------------------------------------------------------------

education_order = (
    df["education"]
    .value_counts()
    .sort_values()
)

plt.figure(figsize=(10, 7))

education_order.plot(
    kind="barh"
)

plt.title(
    "Distribution of Educational Attainment",
    fontsize=15
)

plt.xlabel("Number of Individuals")
plt.ylabel("Education")

plt.tight_layout()

plt.savefig(
    "04_education_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 10. EDUCATION VS INCOME
# ------------------------------------------------------------

education_income = pd.crosstab(
    df["education"],
    df["income"],
    normalize="index"
) * 100

education_income = education_income.sort_values(
    ">50K",
    ascending=True
)

print("\nIncome percentage by education:")
print(education_income)


# ------------------------------------------------------------
# VISUALIZATION 5
# Education vs Income
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

education_income.plot(
    kind="barh",
    stacked=True,
    figsize=(10, 8)
)

plt.title(
    "Income-Class Composition by Education",
    fontsize=15
)

plt.xlabel("Percentage of Individuals")
plt.ylabel("Education")

plt.legend(
    title="Income"
)

plt.tight_layout()

plt.savefig(
    "05_education_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 11. HOURS PER WEEK ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HOURS PER WEEK ANALYSIS")
print("=" * 70)

print(
    df["hours_per_week"].describe()
)


# ------------------------------------------------------------
# VISUALIZATION 6
# Hours Distribution
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="hours_per_week",
    bins=30,
    kde=True
)

plt.title(
    "Distribution of Hours Worked per Week",
    fontsize=15
)

plt.xlabel("Hours Worked per Week")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "06_hours_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# VISUALIZATION 7
# Hours vs Income
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="income",
    y="hours_per_week"
)

plt.title(
    "Hours Worked per Week by Income Class",
    fontsize=15
)

plt.xlabel("Income Class")
plt.ylabel("Hours per Week")

plt.tight_layout()

plt.savefig(
    "07_hours_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 12. WORKCLASS ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WORKCLASS ANALYSIS")
print("=" * 70)

print(
    df["workclass"].value_counts()
)


# Workclass vs income percentages

workclass_income = pd.crosstab(
    df["workclass"],
    df["income"],
    normalize="index"
) * 100

print("\nIncome percentage by workclass:")
print(workclass_income)


# ------------------------------------------------------------
# VISUALIZATION 8
# Workclass vs Income
# ------------------------------------------------------------

workclass_high_income = (
    workclass_income[">50K"]
    .sort_values()
)

plt.figure(figsize=(10, 6))

workclass_high_income.plot(
    kind="barh"
)

plt.title(
    "Percentage Earning >50K by Workclass",
    fontsize=15
)

plt.xlabel("Percentage Earning >50K")
plt.ylabel("Workclass")

plt.tight_layout()

plt.savefig(
    "08_workclass_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 13. OCCUPATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OCCUPATION ANALYSIS")
print("=" * 70)

occupation_counts = (
    df["occupation"]
    .value_counts()
)

print(occupation_counts)


# ------------------------------------------------------------
# VISUALIZATION 9
# Occupation Distribution
# ------------------------------------------------------------

top_occupations = (
    df["occupation"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(10, 7))

sns.barplot(
    x=top_occupations.values,
    y=top_occupations.index
)

plt.title(
    "Most Common Occupation Categories",
    fontsize=15
)

plt.xlabel("Number of Individuals")
plt.ylabel("Occupation")

plt.tight_layout()

plt.savefig(
    "09_occupation_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 14. MARITAL STATUS ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MARITAL STATUS ANALYSIS")
print("=" * 70)

marital_income = pd.crosstab(
    df["marital_status"],
    df["income"],
    normalize="index"
) * 100

print(
    marital_income
)


# ------------------------------------------------------------
# VISUALIZATION 10
# Marital Status vs Income
# ------------------------------------------------------------

marital_high_income = (
    marital_income[">50K"]
    .sort_values()
)

plt.figure(figsize=(10, 6))

marital_high_income.plot(
    kind="barh"
)

plt.title(
    "Percentage Earning >50K by Marital Status",
    fontsize=15
)

plt.xlabel("Percentage Earning >50K")
plt.ylabel("Marital Status")

plt.tight_layout()

plt.savefig(
    "10_marital_status_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 15. SEX VS INCOME
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SEX VS INCOME")
print("=" * 70)

sex_income = pd.crosstab(
    df["sex"],
    df["income"],
    normalize="index"
) * 100

print(sex_income)


# Visualization

plt.figure(figsize=(7, 5))

sns.barplot(
    data=sex_income.reset_index(),
    x="sex",
    y=">50K"
)

plt.title(
    "Percentage Earning >50K by Sex",
    fontsize=15
)

plt.xlabel("Sex")
plt.ylabel("Percentage Earning >50K")

plt.tight_layout()

plt.savefig(
    "11_sex_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 16. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

numeric_df = df.select_dtypes(
    include=np.number
)

correlation_matrix = (
    numeric_df.corr()
)

print(correlation_matrix)


# ------------------------------------------------------------
# VISUALIZATION 12
# CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Heatmap of Numerical Variables",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "12_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 17. MULTIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MULTIVARIATE ANALYSIS")
print("=" * 70)

# Use a reproducible sample for clearer visualization
sample_size = min(10000, len(df))

sample_df = df.sample(
    sample_size,
    random_state=42
)


# ------------------------------------------------------------
# VISUALIZATION 13
# Age + Education + Income
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_df,
    x="age",
    y="education_num",
    hue="income",
    alpha=0.5
)

plt.title(
    "Age, Education Level and Income Class",
    fontsize=15
)

plt.xlabel("Age")
plt.ylabel("Education Number")

plt.legend(
    title="Income"
)

plt.tight_layout()

plt.savefig(
    "13_multivariate_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 18. GROUPED SUMMARY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDUCATION GROUP SUMMARY")
print("=" * 70)

education_summary = (
    df.groupby("education")
    .agg(
        observations=("income", "size"),
        high_income_percentage=(
            "income",
            lambda x: (
                x == ">50K"
            ).mean() * 100
        ),
        median_age=("age", "median"),
        median_hours=(
            "hours_per_week",
            "median"
        )
    )
    .sort_values(
        "high_income_percentage",
        ascending=False
    )
)

print(education_summary)


# ------------------------------------------------------------
# 19. AGE GROUP ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AGE GROUP ANALYSIS")
print("=" * 70)

# Create age groups

df["age_group"] = pd.cut(
    df["age"],
    bins=[
        0,
        20,
        30,
        40,
        50,
        60,
        70,
        100
    ],
    labels=[
        "Under 20",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70+"
    ]
)

age_income = pd.crosstab(
    df["age_group"],
    df["income"],
    normalize="index"
) * 100

print(age_income)


# ------------------------------------------------------------
# VISUALIZATION 14
# Age Group vs Income
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

age_income[">50K"].plot(
    kind="bar"
)

plt.title(
    "Percentage Earning >50K by Age Group",
    fontsize=15
)

plt.xlabel("Age Group")
plt.ylabel("Percentage Earning >50K")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "14_age_group_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 20. OUTLIER VISUALIZATION
# ------------------------------------------------------------

numeric_columns = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week"
]

print("\n" + "=" * 70)
print("OUTLIER SUMMARY")
print("=" * 70)

outlier_results = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    count = outliers.sum()

    percentage = (
        count / len(df) * 100
    )

    outlier_results.append({
        "Variable": column,
        "Outlier Count": count,
        "Outlier Percentage": round(
            percentage,
            2
        )
    })

outlier_report = pd.DataFrame(
    outlier_results
)

print(outlier_report)


# ------------------------------------------------------------
# 21. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)

print("Final dataset shape:", df.shape)

print("\nIncome distribution:")
print(df["income"].value_counts())

print("\nMost common education:")
print(df["education"].value_counts().head())

print("\nMost common occupation:")
print(df["occupation"].value_counts().head())

print("\nMedian age:")
print(df["age"].median())

print("\nMedian hours per week:")
print(df["hours_per_week"].median())

print("\nAll EDA visualizations have been generated.")

print("=" * 70)
