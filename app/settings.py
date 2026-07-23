import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    openai_generation_model: str = os.getenv(
        "OPENAI_GENERATION_MODEL",
        "gpt-4o-mini",
    )
    conversation_history_limit: int = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6"))


settings = Settings()
