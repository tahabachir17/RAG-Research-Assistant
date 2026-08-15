"""Public request and response contracts for the showcase API."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    use_rerank: bool = False
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class SourceChunk(BaseModel):
    chunk_id: str
    paper_id: str
    title: str
    section: str
    citation_number: int
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieved_chunks: list[str]
    citations_valid: bool
    latency_ms: int
    used_rerank: bool
    named_papers: list[str] = Field(default_factory=list)
    papers_without_evidence: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    retriever_ready: bool
    llm_ready: bool


class ErrorResponse(BaseModel):
    detail: str
