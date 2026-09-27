from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POLICY_", env_file=".env", extra="ignore", env_parse_none_str="null"
    )

    corpus_path: Path = Path("data/corpus.json")
    data_path: Path = Path("data/policies")
    index_path: Path = Path("data/index")
    session_path: Path = Path("data/sessions.sqlite")
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "gemma4:26b"
    llm_thinking: bool | None = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model_device: str = "cpu"
    chunk_tokens: int = Field(default=380, ge=100, le=1000)
    overlap_tokens: int = Field(default=45, ge=0)
    candidates: int = Field(default=15, ge=5, le=50)
    rerank_candidates: int = Field(default=20, ge=5, le=60)
    context_chunks: int = Field(default=6, ge=4, le=12)
    min_rerank_score: float = 0.0
    llm_timeout: float = Field(default=180, gt=0)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.overlap_tokens >= self.chunk_tokens / 2:
            raise ValueError("Overlap must be less than half the chunk size.")
        return self
