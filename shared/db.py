import os
from sqlalchemy import create_engine


def get_engine():
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "equipment")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")
