import pandas as pd
from sqlalchemy import create_engine
import os

def handle_null_values_imputation(df):
    """
    Handles null values in the DataFrame. Imputes string columns with 'NoValue' and fills numeric columns with 0.
    """
    for column in df.columns:
        if df[column].dtype == 'object':  # For string columns
            df[column].fillna('NoValue', inplace=True)
        else:  # For numeric columns
            df[column].fillna(0, inplace=True)
    return df

def remove_unneeded_fields(df, columns_to_remove):
    """
    Removes specified columns from the DataFrame.
    """
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    return df

def extract_data_from_db():
    """
    Extracts data from MySQL database.
    """
    mysql_host = "localhost"
    mysql_user = "root"
    mysql_password = ""
    mysql_db = "superstore"

    engine = create_engine(
        f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
    )

    query = "SELECT * FROM superstore_data"
    df = pd.read_sql(query, con=engine)
    return df
