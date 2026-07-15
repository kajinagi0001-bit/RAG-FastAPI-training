from sqlalchemy import text

from app.database import engine


def ensure_schema() -> None:
    with engine.begin() as connection:
        existing_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(chunks)")).fetchall()
        }
        if "embedding_json" not in existing_columns:
            connection.execute(text("ALTER TABLE chunks ADD COLUMN embedding_json TEXT"))
        if "embedding_model" not in existing_columns:
            connection.execute(text("ALTER TABLE chunks ADD COLUMN embedding_model VARCHAR(100)"))
