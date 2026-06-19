import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("sales_data.csv")
print("===== ORIGINAL DATA =====")
print(df)
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())
df["Sales"] = df["Sales"].fillna(df["Sales"].mean())
df = df.drop_duplicates()
Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
df = df[(df["Sales"] >= lower_limit) & (df["Sales"] <= upper_limit)]
print("\n===== CLEANED DATA =====")
print(df)
df.to_csv("cleaned_sales_data.csv", index=False)
print("\nCleaned data saved as 'cleaned_sales_data.csv'")
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Category", y="Sales")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], bins=10, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(6, 4))
sns.heatmap(df.select_dtypes(include=np.number).corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()