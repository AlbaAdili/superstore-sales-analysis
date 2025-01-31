import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from database import load_data_to_mysql
from services import extract_data_from_db, handle_null_values_imputation, remove_unneeded_fields

# File path for dataset
file_path = "superstore-data.csv"

# Load data from CSV
df = pd.read_csv(file_path)

# Handle missing values
df = handle_null_values_imputation(df)

# Remove unnecessary columns
columns_to_remove = ["Postal Code"]
df = remove_unneeded_fields(df, columns_to_remove)

# Load data into MySQL
load_data_to_mysql(file_path, table_name="superstore_data")

# Extract data from MySQL
db_data = extract_data_from_db()

# 📌 Descriptive Statistics Heatmap
print("\nDescriptive Statistics:")
print(df.describe())

plt.figure(figsize=(12, 6))
sns.heatmap(df.describe(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Descriptive Statistics Heatmap")
plt.show()

# 📌 Total Sales by Category
grouped_sales = df.groupby("Category")["Sales"].sum()
print("\nTotal Sales by Category:")
print(grouped_sales)

plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_sales.index, y=grouped_sales.values, palette="coolwarm")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# 📌 Total Sales by Region (NEW!)
grouped_sales_region = df.groupby("Region")["Sales"].sum()
print("\nTotal Sales by Region:")
print(grouped_sales_region)

plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_sales_region.index, y=grouped_sales_region.values, palette="Greens")
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# 📌 Correlation Matrix Heatmap
numeric_data = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# 📌 Sales vs. Profit Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Sales"], y=df["Profit"], hue=df["Category"], palette="viridis")
plt.title("Sales vs. Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()

# 📌 Profit Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Profit"], bins=30, kde=True, color="blue")
plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show(block=True)

