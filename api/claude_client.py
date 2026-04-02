"""
Claude API client for querying documents via the Anthropic Python SDK.

This module initialises the Anthropic client from the ANTHROPIC_API_KEY
environment variable and implements an agentic tool-use loop that lets
Claude progressively discover and read documents to answer user queries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from api.tools import TOOLS, execute_tool

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 5

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful research assistant. You answer questions using a "
    "collection of text documents.\n\n"
    "Your workflow:\n"
    "1. Call find_files with search terms derived from the query — include "
    "names, topics, and keyword variations. The server scores files by "
    "matching your terms against metadata-rich filenames (2x weight) and "
    "file content (1x weight), then returns ranked results.\n"
    "2. Call read_files to batch-read the top-scoring files.\n"
    "3. Answer based ONLY on the documents. Cite filenames.\n\n"
    "Tips: generate broad search terms for better recall. Include person "
    "names, topics, synonyms, and related terms. You can call find_files "
    "again with different terms if the first pass misses relevant documents. "
    "Use search_files only as a fallback for exact phrase matching."
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class APIKeyMissingError(Exception):
    """Raised when the ANTHROPIC_API_KEY environment variable is not set."""


class ClaudeAPIError(Exception):
    """Raised when the Claude API returns an error."""


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------


def get_client() -> anthropic.Anthropic:
    """Return an initialised Anthropic client.

    Raises
    ------
    APIKeyMissingError
        If the ``ANTHROPIC_API_KEY`` environment variable is not set or is
        empty.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise APIKeyMissingError(
            "The ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it to your Anthropic API key before making requests."
        )
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


@dataclass
class QueryStats:
    """Tracks usage statistics for a single query."""

    api_calls: int = 0
    tool_rounds: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0

    def record_api_call(self, response: anthropic.types.Message) -> None:
        self.api_calls += 1
        if hasattr(response, "usage") and response.usage:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "api_calls": self.api_calls,
            "tool_rounds": self.tool_rounds,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


# ---------------------------------------------------------------------------
# Agentic tool-use loop
# ---------------------------------------------------------------------------


def _call_api(
    client: anthropic.Anthropic,
    messages: list[dict],
    model: str,
    max_tokens: int,
    *,
    include_tools: bool = True,
) -> anthropic.types.Message:
    """Make a single API call with error handling.

    Raises
    ------
    ClaudeAPIError
        If the API call fails (authentication, rate limit, timeout, etc.).
    """
    try:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        if include_tools:
            kwargs["tools"] = TOOLS
        return client.messages.create(**kwargs)
    except anthropic.AuthenticationError as exc:
        raise ClaudeAPIError(
            "Authentication failed. Please verify your ANTHROPIC_API_KEY is "
            f"correct and active. Details: {exc}"
        ) from exc
    except anthropic.RateLimitError as exc:
        raise ClaudeAPIError(
            "Rate limit exceeded. Please wait a moment and try again. "
            f"Details: {exc}"
        ) from exc
    except anthropic.APITimeoutError as exc:
        raise ClaudeAPIError(
            f"Request to Claude timed out. Please try again. Details: {exc}"
        ) from exc
    except anthropic.APIError as exc:
        raise ClaudeAPIError(
            f"Claude API error: {exc}"
        ) from exc


def _extract_text(response: anthropic.types.Message) -> str:
    """Extract concatenated text from a Claude response."""
    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts)


def ask_claude(
    query: str,
    folder_path: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, list[str], QueryStats]:
    """Run an agentic tool-use loop to answer *query* using documents in *folder_path*.

    Claude generates search terms, the server scores files using
    deterministic glob+grep, and Claude reads the top hits to answer.

    Returns
    -------
    tuple[str, list[str], QueryStats]
        A ``(answer, sources, stats)`` tuple.
    """
    client = get_client()
    sources: list[str] = []
    stats = QueryStats()

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Document folder: {folder_path}"
            ),
        }
    ]

    for _round_num in range(MAX_TOOL_ROUNDS):
        response = _call_api(client, messages, model, max_tokens)
        stats.record_api_call(response)

        if response.stop_reason == "tool_use":
            stats.tool_rounds += 1
            assistant_content = response.content
            tool_results = []

            for block in assistant_content:
                if block.type == "tool_use":
                    stats.record_tool_call(block.name)
                    result = execute_tool(block.name, block.input)

                    # Track sources from read calls
                    if block.name == "read_files":
                        for fp in block.input.get("file_paths", []):
                            fname = Path(fp).name
                            if fname not in sources:
                                sources.append(fname)
                    elif block.name == "read_file":
                        fname = Path(block.input["file_path"]).name
                        if fname not in sources:
                            sources.append(fname)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            return _extract_text(response), sources, stats

    # Safety limit reached — force a text-only answer
    messages.append(
        {
            "role": "user",
            "content": (
                "You've reached the maximum number of tool calls. Please provide "
                "your best answer based on the information you've gathered so far."
            ),
        }
    )

    response = _call_api(
        client, messages, model, max_tokens, include_tools=False
    )
    stats.record_api_call(response)
    return _extract_text(response), sources, stats
