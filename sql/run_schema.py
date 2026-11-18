import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load variables from .env file
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise ValueError("DATABASE_URL is not set in the .env file!")

# Initialize SQLAlchemy engine
engine = create_engine(DB_URL)

# Read schema.sql
with open("sql/schema.sql", "r") as f:
    schema_sql = f.read()

# Execute schema on Neon database
with engine.connect() as conn:
    conn.execute(text(schema_sql))
    conn.commit()
    print("Schema successfully created on Neon Cloud PostgreSQL!")