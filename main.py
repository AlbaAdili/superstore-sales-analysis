import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from database import load_data_to_mysql
from services import extract_data_from_db, handle_null_values_imputation, remove_unneeded_fields
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

#  Descriptive Statistics Heatmap
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

#  CUSTOMER SEGMENTATION (K-Means Clustering)
customer_data = df.groupby("Customer ID")[["Sales", "Profit"]].sum().reset_index()

# Standardize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[["Sales", "Profit"]])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_data["Cluster"] = kmeans.fit_predict(customer_data_scaled)

#  Visualizing Customer Segments
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_data["Sales"], y=customer_data["Profit"], hue=customer_data["Cluster"], palette="viridis")
plt.title("Customer Segmentation (K-Means Clustering)")
plt.xlabel("Total Sales")
plt.ylabel("Total Profit")
plt.show()

#  BEST TIME TO OFFER DISCOUNTS (Discount vs. Profit Trend)
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["YearMonth"] = df["Order Date"].dt.to_period("M")  # Extract Year-Month

# Aggregate discount trends per month
monthly_discount = df.groupby("YearMonth")[["Discount", "Profit"]].mean().reset_index()
monthly_discount["YearMonth"] = monthly_discount["YearMonth"].astype(str)  # Convert to string for plotting

#  Plot Discount Trends
fig, ax1 = plt.subplots(figsize=(12, 6))

ax2 = ax1.twinx()
ax1.plot(monthly_discount["YearMonth"], monthly_discount["Discount"], 'g-', marker='o', label="Avg Discount (%)")
ax2.plot(monthly_discount["YearMonth"], monthly_discount["Profit"], marker='o', linestyle="dashed", color="blue", label="Avg Profit")


ax1.set_xlabel("Year-Month")
ax1.set_ylabel("Average Discount (%)", color='g')
ax2.set_ylabel("Average Profit", color='b')

plt.title("Best Time to Offer Discounts (Discount vs. Profit Trend)")
fig.autofmt_xdate()
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()

#  SALES FORECAST (Predict Sales for the Next 6 Months)
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

#  Plot Sales Forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales["YearMonth"], monthly_sales["Sales"], marker='o', label="Actual Sales")
plt.plot(pd.date_range(start=monthly_sales["YearMonth"].iloc[-1], periods=6, freq="M").strftime('%Y-%m'),
         future_sales, marker='o', linestyle="dashed", color="red", label="Predicted Sales")

plt.xticks(rotation=45)
plt.xlabel("Year-Month")
plt.ylabel("Total Sales")
plt.title("Sales Forecast for Next 6 Months")
plt.legend()
plt.show()

