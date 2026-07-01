import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Titanic-Dataset.csv")
print(df.describe())
print(df.isnull().sum())
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
plt.boxplot(df["Fare"].dropna())
plt.title("Fare Boxplot")
plt.show()
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()
sns.pairplot(df[["Age", "Fare", "Survived"]].dropna())
plt.show()