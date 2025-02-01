# **Superstore Sales Analysis: Data-Driven Insights & Forecasting**

## Project Overview
This project aims to analyze **Superstore Sales Data** using **Python and MySQL** to extract business insights, predict future sales, and evaluate the impact of discounts on profitability. The analysis includes **data cleaning, visualization, machine learning models, and database integration**.  

### Key Objectives
- **Data Cleaning & Preprocessing**: Handling missing values, duplicates, and structuring data.  
- **Exploratory Data Analysis (EDA)**: Sales trends, profitability analysis, and regional insights.  
- **Machine Learning Models**:
  - **Sales Forecasting (Linear Regression)**
  - **Customer Segmentation (K-Means Clustering)**
- **Database Integration**: Storing and retrieving data using **MySQL**.  

## Technologies Used
- **Programming Language**: Python   
- **Database**: MySQL   
- **Visualization Libraries**: Matplotlib, Seaborn   
- **Machine Learning**: Scikit-Learn   
- **Database Connection**: SQLAlchemy  


## Step-by-Step Guide
1. Clone the Repository
````
- git clone <https://github.com/AlbaAdili/superstore-sales-analysis.git>
- cd superstore-sales-analysis
````
2. Install Dependencies
````
pip install -r requirements.txt
````
3. Set Up MySQL Database
````
sudo systemctl start mysql  # macOS/Linux
net start mysql             # Windows
````
4. Create a Database
````
CREATE DATABASE superstore;
````
5. Run the Analysis & Generate Visualizations
````
python main.py
````


### MySQL Database Schema
````sql
CREATE TABLE superstore_data (
    Row_ID INT PRIMARY KEY,
    Order_ID VARCHAR(50),
    Order_Date DATE,
    Ship_Date DATE,
    Ship_Mode VARCHAR(50),
    Customer_ID VARCHAR(50),
    Customer_Name VARCHAR(255),
    Segment VARCHAR(50),
    Country VARCHAR(50),
    City VARCHAR(50),
    State VARCHAR(50),
    Region VARCHAR(50),
    Product_ID VARCHAR(50),
    Category VARCHAR(50),
    Sub_Category VARCHAR(50),
    Product_Name VARCHAR(255),
    Sales FLOAT,
    Quantity INT,
    Discount FLOAT,
    Profit FLOAT
);
