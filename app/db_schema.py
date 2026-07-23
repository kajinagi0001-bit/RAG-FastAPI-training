from sqlalchemy import text

from app.database import engine


def ensure_schema() -> None:
    with engine.begin() as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).fetchall()
        }
        if "chunks" not in table_names:
            return

        chunk_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(chunks)")).fetchall()
        }
        if "embedding_json" not in chunk_columns:
            connection.execute(text("ALTER TABLE chunks ADD COLUMN embedding_json TEXT"))
        if "embedding_model" not in chunk_columns:
            connection.execute(text("ALTER TABLE chunks ADD COLUMN embedding_model VARCHAR(100)"))

        if "rag_runs" not in table_names:
            return

        rag_run_columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info(rag_runs)")).fetchall()
        }
        if "run_type" not in rag_run_columns:
            connection.execute(
                text("ALTER TABLE rag_runs ADD COLUMN run_type VARCHAR(50) DEFAULT 'unknown' NOT NULL")
            )
