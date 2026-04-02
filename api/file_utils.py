"""
File discovery and pre-filtering utilities using glob + grep strategies.

Mirrors the Manus context-engineering principle of using glob (filename
pattern matching) and grep (content search) for progressive disclosure —
only loading file contents into context when they are likely relevant.

Filenames are expected to follow the naming convention produced by
extract_pages.py:

    page_NNN_<entity1>_<entity2>_<topic1>_<topic2>.txt

where entities are proper nouns (people, organisations, locations) and topics
are frequent nouns extracted via spaCy NER.  These descriptive names act as
lightweight metadata so that glob-style matching can identify candidate files
before their content is read.

Scoring combines two signals:
  - Glob score  (weight ×2): query terms found in the filename tokens
  - Grep score  (weight ×1): query terms found in the file content
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class FileResult(TypedDict):
    """A single file entry returned by the filtering functions."""
    path: str
    filename: str
    score: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches the leading page_NNN prefix (one or more digits after "page_").
_PAGE_PREFIX_RE = re.compile(r"^page_\d+_?")

# Characters used to split query text into individual terms.
_QUERY_SPLIT_RE = re.compile(r"[\s,;:!?\-/\.\(\)\[\]\{\}\"\']+")


def _tokenize_filename(filename: str) -> set[str]:
    """Extract lowercase tokens from a filename, excluding the page_NNN prefix
    and the .txt extension.

    Example
    -------
    >>> _tokenize_filename("page_077_ariana_arellano_jarelle_cenidoza_s_molly_chuen_durchweiler_people_overtime_employee.txt")
    {'ariana', 'arellano', 'jarelle', 'cenidoza', 's', 'molly', 'chuen', 'durchweiler', 'people', 'overtime', 'employee'}
    """
    # Strip extension
    stem = Path(filename).stem  # e.g. "page_077_ariana_arellano_..."

    # Remove the leading page_NNN prefix
    stem = _PAGE_PREFIX_RE.sub("", stem)

    # Split on underscores, lowercase, and drop empty / purely-numeric tokens
    tokens: set[str] = set()
    for part in stem.split("_"):
        part = part.lower().strip()
        if part and not part.isdigit():
            tokens.add(part)
    return tokens


def _grep_content(filepath: Path, query_terms: set[str], char_limit: int = 5000) -> int:
    """Count how many *query_terms* appear in the file content (grep-style).

    Only the first *char_limit* characters are read to keep latency bounded
    across large collections.  Matching is case-insensitive.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0

    if len(text) > char_limit:
        text = text[:char_limit]

    text_lower = text.lower()
    return sum(1 for term in query_terms if term in text_lower)


def _tokenize_query(query: str) -> set[str]:
    """Split a user query into a set of lowercase search terms.

    Splits on whitespace and common punctuation.  Drops empty strings and
    purely-numeric tokens so that page numbers and dates don't cause
    false-positive matches.

    Example
    -------
    >>> sorted(_tokenize_query("What overtime did Ariana work?"))
    ['ariana', 'did', 'overtime', 'what', 'work']
    """
    terms: set[str] = set()
    for raw in _QUERY_SPLIT_RE.split(query):
        term = raw.lower().strip()
        if term and not term.isdigit():
            terms.add(term)
    return terms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_text_files(folder_path: str | Path) -> list[Path]:
    """Return a sorted list of all ``.txt`` file paths inside *folder_path*.

    Parameters
    ----------
    folder_path:
        Path to the directory to scan.

    Returns
    -------
    list[Path]
        Sorted list of ``Path`` objects pointing to .txt files.

    Raises
    ------
    FileNotFoundError
        If *folder_path* does not exist.
    NotADirectoryError
        If *folder_path* exists but is not a directory.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    return sorted(folder.glob("*.txt"))


def filter_files_by_query(
    files: list[Path],
    query: str,
    max_files: int = 10,
) -> list[FileResult]:
    """Score and rank *files* using glob (filename) and grep (content) signals.

    Scoring
    -------
    Each file receives a combined score:

    * **Glob score** (×2): number of query terms found in the filename tokens.
      Filename tokens are derived by stripping the ``page_NNN`` prefix and
      ``.txt`` extension and splitting on underscores.  These curated names
      encode entities and topics, so a filename match is high-confidence.

    * **Grep score** (×1): number of query terms found anywhere in the file
      content (case-insensitive, first 5 000 chars).  This catches relevant
      files whose names don't happen to contain the exact query terms.

    ``combined = glob_score * 2 + grep_score``

    Fallback behaviour
    ------------------
    If **no** file scores above 0, the first *max_files* files are returned
    (sorted by filename) so that the downstream query still receives context.

    Parameters
    ----------
    files:
        List of ``Path`` objects (typically from :func:`list_text_files`).
    query:
        The user's natural-language query string.
    max_files:
        Maximum number of files to return.  Defaults to ``10``.

    Returns
    -------
    list[FileResult]
        A list of dicts with keys ``path``, ``filename``, and ``score``,
        sorted by combined score descending (then filename ascending for ties).
    """
    query_terms = _tokenize_query(query)

    scored: list[FileResult] = []
    for filepath in files:
        filename = filepath.name

        # Glob signal: match query terms against filename tokens
        filename_tokens = _tokenize_filename(filename)
        glob_score = len(query_terms & filename_tokens)

        # Grep signal: search file content for query terms
        grep_score = _grep_content(filepath, query_terms)

        combined = glob_score * 2 + grep_score
        scored.append(
            FileResult(path=str(filepath), filename=filename, score=combined)
        )

    # Check whether any file matched at all
    has_matches = any(entry["score"] > 0 for entry in scored)

    if has_matches:
        # Sort by score descending, then filename ascending for determinism
        scored.sort(key=lambda e: (-e["score"], e["filename"]))
    else:
        # Fallback: return the first max_files files in filename order
        scored.sort(key=lambda e: e["filename"])

    return scored[:max_files]
