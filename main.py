import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from database import load_data_to_mysql
from services import extract_data_from_db, handle_null_values_imputation, remove_unneeded_fields
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#  File path for dataset
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

# Descriptive Statistics Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.describe(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Descriptive Statistics Heatmap")
plt.show()

#  Total Sales by Category
grouped_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_sales.index, y=grouped_sales.values, hue=grouped_sales.index, palette="coolwarm", legend=False)
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

#  Total Sales by Region
grouped_sales_region = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(10, 6))
sns.barplot(x=grouped_sales_region.index, y=grouped_sales_region.values, hue=grouped_sales_region.index, palette="Greens", legend=False)
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

#  Correlation Matrix Heatmap
numeric_data = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

#  Sales vs. Profit Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Sales"], y=df["Profit"], hue=df["Category"], palette="viridis")
plt.title("Sales vs. Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()

#  Profit Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Profit"], bins=30, kde=True, color="blue")
plt.title("Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show()

#  SALES FORECAST (Predict Sales for the Next 6 Months)
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["YearMonth"] = df["Order Date"].dt.to_period("M")  # Extract Year-Month

# Group sales by Year-Month
monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()
monthly_sales["YearMonth"] = monthly_sales["YearMonth"].astype(str)  # Convert to string for plotting

# Convert Year-Month to numeric values for model training
monthly_sales["NumericDate"] = range(len(monthly_sales))

#  Train Linear Regression Model
X = monthly_sales[["NumericDate"]]
y = monthly_sales["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

#  Predict Future Sales
future_dates = np.array(range(len(monthly_sales), len(monthly_sales) + 6)).reshape(-1, 1)
future_sales = model.predict(pd.DataFrame(future_dates, columns=["NumericDate"]))

# Fix: Ensure x and y dimensions match
future_x_labels = list(pd.date_range(start=monthly_sales["YearMonth"].iloc[-1], periods=6, freq="M").strftime('%Y-%m'))

#  Plot Sales Forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales["YearMonth"], monthly_sales["Sales"], marker='o', label="Actual Sales")
plt.plot(future_x_labels, future_sales, marker='o', linestyle="dashed", color="red", label="Predicted Sales")

plt.xticks(rotation=45)
plt.xlabel("Year-Month")
plt.ylabel("Total Sales")
plt.title("Sales Forecast for Next 6 Months")
plt.legend()
plt.show()

#  MOST PROFITABLE CATEGORY PREDICTION (Machine Learning Model)
df["Category"] = LabelEncoder().fit_transform(df["Category"])  # Convert category to numbers
df["Sub-Category"] = LabelEncoder().fit_transform(df["Sub-Category"])

X = df[["Sales", "Quantity", "Discount"]]  # Features
y = df["Category"]  # Target (Category)

#  Train Decision Tree Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#  Make Prediction
predicted_category = model.predict(pd.DataFrame([[500, 2, 0.1]], columns=X.columns))

#  Plot Predicted Category
plt.figure(figsize=(8, 5))
sns.barplot(x=list(df["Category"].unique()), y=[1 if cat == predicted_category[0] else 0 for cat in df["Category"].unique()], hue=list(df["Category"].unique()), palette="coolwarm", legend=False)
plt.title("Predicted Most Profitable Category")
plt.xlabel("Category")
plt.ylabel("Prediction Confidence")
plt.show()

print(f"\nPredicted Most Profitable Category for Sales=500, Quantity=2, Discount=10%: {df['Category'].unique()[predicted_category[0]]}")

