"""Tool definitions for Claude tool-use API.

Defines find_files, search_files, and read_files tools.  Claude generates
smart search terms; the server does fast deterministic glob+grep scoring
to rank files; Claude reads the top hits and answers.
"""

from __future__ import annotations

from pathlib import Path

from api.file_utils import list_text_files, filter_files_by_query

TOOLS = [
    {
        "name": "find_files",
        "description": (
            "Find the most relevant document files by scoring them against "
            "your search terms. Each filename encodes entities and topics "
            "extracted via NLP (e.g., page_077_ariana_arellano_overtime_employee.txt), "
            "so search terms are matched against BOTH filename metadata (weighted 2x) "
            "AND file content (weighted 1x). Returns ranked results with scores.\n\n"
            "Tips for generating good search terms:\n"
            "- Use entity names: 'ariana', 'arellano', 'cenidoza'\n"
            "- Use topic keywords: 'overtime', 'diabetes', 'supervisor'\n"
            "- Try variations: names may be truncated in filenames\n"
            "- Cast a wide net: more terms = better recall"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {
                    "type": "string",
                    "description": "Absolute path to the document folder.",
                },
                "search_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of search terms to match against filenames "
                        "and content. e.g., ['ariana', 'overtime', 'hours']"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max files to return (default 10).",
                },
            },
            "required": ["folder_path", "search_terms"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Grep-style content search: find files containing a specific "
            "term (case-insensitive). Returns matching filenames with brief "
            "context snippets. Use this as a fallback when find_files didn't "
            "surface the right documents — e.g., to search for exact phrases "
            "or terms not in filenames."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {
                    "type": "string",
                    "description": "Absolute path to the directory to search.",
                },
                "search_term": {
                    "type": "string",
                    "description": "Term to search for (case-insensitive).",
                },
            },
            "required": ["folder_path", "search_term"],
        },
    },
    {
        "name": "read_files",
        "description": (
            "Read the full content of one or more files. Returns each file's "
            "text, truncated at 5000 characters per file. Accepts up to 10 "
            "file paths at once — prefer batching over multiple single reads."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of absolute file paths to read (max 10)."
                    ),
                },
            },
            "required": ["file_paths"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool handler functions
# ---------------------------------------------------------------------------


def handle_find_files(
    folder_path: str,
    search_terms: list[str],
    max_results: int = 10,
) -> str:
    """Score and rank files using glob+grep against Claude's search terms.

    Wraps the deterministic filter_files_by_query pipeline: Claude provides
    smart terms, the server does fast filename + content matching.
    """
    try:
        all_files = list_text_files(folder_path)
        if not all_files:
            return "No .txt files found in the folder."

        # Join terms into a query string for the existing scoring function
        query_str = " ".join(search_terms)
        results = filter_files_by_query(all_files, query_str, max_files=max_results)

        if not results:
            return "No matching files found."

        lines: list[str] = []
        for entry in results:
            lines.append(f"[score={entry['score']}] {entry['path']}")

        return "\n".join(lines)
    except (FileNotFoundError, NotADirectoryError) as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error finding files: {exc}"


def handle_search_files(folder_path: str, search_term: str) -> str:
    """Search all .txt files in a directory for a term (case-insensitive).

    Returns matching filenames with brief context snippets.
    """
    try:
        folder = Path(folder_path)
        if not folder.exists():
            return f"Error: folder does not exist: {folder_path}"
        if not folder.is_dir():
            return f"Error: path is not a directory: {folder_path}"

        term_lower = search_term.lower()
        matches: list[str] = []

        for txt_file in sorted(folder.glob("*.txt")):
            if not txt_file.is_file():
                continue
            try:
                content = txt_file.read_text(encoding="utf-8", errors="replace")[:5000]
            except OSError:
                continue

            if term_lower in content.lower():
                # Find the first line containing the match for context
                snippet = ""
                for line in content.splitlines():
                    if term_lower in line.lower():
                        snippet = line.strip()[:200]
                        break
                matches.append(f"{txt_file.name}:\n  {snippet}")

        if not matches:
            return f"No files contain '{search_term}'"

        header = f"Found in {len(matches)} file{'s' if len(matches) != 1 else ''}:"
        return header + "\n\n" + "\n\n".join(matches)
    except Exception as exc:
        return f"Error searching files: {exc}"


def handle_read_file(file_path: str) -> str:
    """Read the full content of a file, truncated at 5000 characters."""
    try:
        p = Path(file_path)
        if not p.exists():
            return f"Error: file not found: {file_path}"
        if not p.is_file():
            return f"Error: path is not a file: {file_path}"

        content = p.read_text(encoding="utf-8", errors="replace")

        if len(content) > 5000:
            content = content[:5000] + "\n\n... [truncated at 5000 characters]"

        return content
    except PermissionError:
        return f"Error: permission denied reading file: {file_path}"
    except Exception as exc:
        return f"Error reading file: {exc}"


def handle_read_files(file_paths: list[str]) -> str:
    """Read multiple files at once, returning each with a header. Max 10."""
    if len(file_paths) > 10:
        file_paths = file_paths[:10]

    sections: list[str] = []
    for fp in file_paths:
        header = f"### {Path(fp).name}"
        content = handle_read_file(fp)
        sections.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Tool map and dispatch
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "find_files": handle_find_files,
    "search_files": handle_search_files,
    "read_file": handle_read_file,
    "read_files": handle_read_files,
}


def execute_tool(
    tool_name: str,
    tool_input: dict,
    *,
    work_dir: Path | None = None,
) -> str:
    """Execute a tool by name with the given input parameters."""
    handler = TOOL_MAP.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"
    return handler(**tool_input)
