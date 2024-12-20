import sqlite3
import pandas as pd

class DataLoader:
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name

    def load_data(self):
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_path)
        # Query data from the specified table
        query = f"SELECT * FROM {self.table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
