"""Environment-backed application settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded once by application components."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )

    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS: int = Field(default=1024, gt=0)
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    ANTHROPIC_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OLLAMA_HOST: str | None = None

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "ai_papers"
    DENSE_TOP_K: int = 50
    SPARSE_TOP_K: int = 50
    HYBRID_TOP_K: int = 20
    RERANK_TOP_K: int = 8
    MAX_CHUNK_TOKENS: int = 512
    CHUNK_OVERLAP: int = 80
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RATE_LIMIT_RPM: int = 60
