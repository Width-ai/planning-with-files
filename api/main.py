"""
FastAPI application for querying descriptively-named text files using Claude.

Entry point: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env file for local development (no-op if file is absent).
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from api.claude_client import ask_claude, APIKeyMissingError, ClaudeAPIError

app = FastAPI(
    title="Planning With Files API",
    description="Query descriptively-named text files using Claude as the reasoning engine.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming request body for the /query endpoint."""

    query: str
    folder_path: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must be a non-empty string")
        return v.strip()

    @field_validator("folder_path")
    @classmethod
    def folder_path_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("folder_path must be a non-empty string")
        return v.strip()


class QueryStatsResponse(BaseModel):
    """Usage statistics for the query."""

    api_calls: int
    tool_rounds: int
    tool_calls: dict[str, int]
    input_tokens: int
    output_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    """Response body returned by the /query endpoint."""

    answer: str
    sources: list[str]
    stats: QueryStatsResponse


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Container / load-balancer health check."""
    return HealthResponse(status="ok")


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request – invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Accept a user query and a folder path, let Claude agentic loop discover
    and read relevant files, and return the answer along with source filenames.
    """

    # --- Input validation beyond Pydantic --------------------------------
    folder = Path(request.folder_path)

    if not folder.exists():
        raise HTTPException(
            status_code=400,
            detail=f"folder_path does not exist: {request.folder_path}",
        )

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"folder_path is not a directory: {request.folder_path}",
        )

    # --- Core logic --------------------------------------------------------
    try:
        answer, sources, stats = ask_claude(request.query, request.folder_path)
        return QueryResponse(
            answer=answer,
            sources=sources,
            stats=QueryStatsResponse(**stats.to_dict()),
        )

    except APIKeyMissingError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except ClaudeAPIError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
