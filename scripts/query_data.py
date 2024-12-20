import pandas as pd
from db_connection import get_sqlalchemy_engine  # Import the function for SQLAlchemy engine

def fetch_data(query: str):
    """Fetch data from the PostgreSQL database using SQLAlchemy."""
    engine = get_sqlalchemy_engine()  # Use SQLAlchemy engine instead of psycopg2 connection
    try:
        # Use the SQLAlchemy engine to fetch data
        df = pd.read_sql(query, engine)
        print("Data fetched successfully.")
        return df
    except Exception as e:
        print(f"Error: Unable to fetch data. {e}")
        return None

if __name__ == "__main__":
    # Example query to test the connection
    query = "SELECT * FROM xdr_data LIMIT 10;"  # Replace 'xdr_data' with your actual table name
    data = fetch_data(query)
    
    if data is not None:
        print(data.head())  # Preview the first few rows
    else:
        print("No data fetched.")
