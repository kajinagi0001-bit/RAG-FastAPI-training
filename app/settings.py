import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER") or (
        "openai" if os.getenv("OPENAI_API_KEY") else "local"
    )
    generation_provider: str = os.getenv("GENERATION_PROVIDER") or (
        "openai" if os.getenv("OPENAI_API_KEY") else "local"
    )
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    openai_generation_model: str = os.getenv(
        "OPENAI_GENERATION_MODEL",
        "gpt-4o-mini",
    )
    local_embedding_dimensions: int = int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "64"))
    conversation_history_limit: int = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6"))


settings = Settings()
