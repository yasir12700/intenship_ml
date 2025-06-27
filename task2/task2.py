# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
df = pd.read_csv("dataset/Titanic-Dataset.csv")



# Dataset Summary
print("Basic Info:")
print(df.info(), "\n")

print("Missing Values:")
print(df.isnull().sum(), "\n")

print("Summary Statistics (Numerical Columns):")
print(df.describe(), "\n")

print("Mean Values:")
print(df.mean(numeric_only=True), "\n")

print("Median Values:")
print(df.median(numeric_only=True), "\n")

print("Standard Deviation:")
print(df.std(numeric_only=True), "\n")

# Histograms
df.hist(bins=20, color='skyblue', edgecolor='black', figsize=(14, 10))
plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()

# Boxplots
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f'Boxplot of {col}')
    plt.show()

# Correlation Matrix & Pairplot
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']].dropna(), hue='Survived', palette='husl')
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.show()

# Identify Patterns, Trends, Anomalies
print("Survival Rate by Pclass:\n", df.groupby('Pclass')['Survived'].mean(), "\n")
print("Survival Rate by Sex:\n", df.groupby('Sex')['Survived'].mean(), "\n")
print("Average Fare by Pclass:\n", df.groupby('Pclass')['Fare'].mean(), "\n")

# Inferences
print("\n--- Key Insights ---")
print("1. Women had a significantly higher survival rate than men.")
print("2. Passengers in 1st class had higher survival chances.")
print("3. Fare is positively correlated with survival.")
print("4. Many passengers have missing 'Age' values.")
print("5. 'Fare' is right-skewed and contains outliers.")