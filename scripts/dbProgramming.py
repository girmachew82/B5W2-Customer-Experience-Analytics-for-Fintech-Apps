import psycopg2
from sqlalchemy import create_engine

class DBProgramming:
    def __init__(self, user, password, host, port, database):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def connection_string(self):
        return f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'

    def insert_dataframe(self, df, table_name):
        try:
            engine = create_engine(self.connection_string())
            df.to_sql(table_name, engine, if_exists='append', index=False)
            print(f"✅ Data inserted into '{table_name}' successfully.")
        except Exception as e:
            print("❌ Insert failed:", e)
