import pandas as pd
from sqlalchemy import create_engine

def load_data_to_mysql(data_file, table_name):
    """
    Load data from CSV file into MySQL database.
    """
    # Load CSV into DataFrame
    data = pd.read_csv(data_file)

    # MySQL connection details
    mysql_host = "localhost"
    mysql_user = "root"
    mysql_password = ""
    mysql_db = "superstore"

    engine = create_engine(
        f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
    )

    # Upload data to MySQL
    data.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"Data loaded successfully into {table_name} table.")
