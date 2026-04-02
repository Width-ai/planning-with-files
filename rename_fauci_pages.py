#!/usr/bin/env python3
"""
rename_fauci_pages.py

Renames existing page_NNN.txt files in fauci_deposition_pages/ to include
NLP-derived entity and topic keywords, matching the naming convention
used for matheson_deposition_pages/.

Usage:
    uv run --with spacy python rename_fauci_pages.py
"""

import os
import re
import subprocess
import sys
from collections import Counter

import spacy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fauci_deposition_pages")
MAX_FILENAME_LENGTH = 200
ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LAW"}
MAX_ENTITIES_IN_NAME = 10
MAX_TOPICS_IN_NAME = 5

# Boilerplate lines that repeat on every page — strip before NLP so they
# don't consume entity/topic slots with redundant info.
_BOILERPLATE_RE = re.compile(
    r"^---\s*Page\s+\d+\s+of\s+\d+\s*---$"          # --- Page X of Y ---
    r"|^DR\.\s*ANTHONY\s+FAUCI\s+\d+/\d+/\d+"        # DR. ANTHONY FAUCI 11/23/2022
    r"|^Page\s+\d+$"                                   # Page 13
    r"|^\d+\s*$"                                       # bare line numbers
    r"|^LEXITAS\s+LEGAL.*$"                            # LEXITAS LEGAL footer
    r"|^www\.lexitaslegal\.com.*$",                    # URL footer line
    re.IGNORECASE | re.MULTILINE,
)
SPACY_MODEL = "en_core_web_sm"


# ---------------------------------------------------------------------------
# Helpers (same logic as extract_pages.py)
# ---------------------------------------------------------------------------

def ensure_spacy_model(model_name: str) -> spacy.language.Language:
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model '{model_name}' ...")
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name],
        )
        nlp = spacy.load(model_name)
    return nlp


def sanitize_token(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9_]", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text.lower()


def extract_entities(doc: spacy.tokens.Doc) -> list[str]:
    seen = set()
    entities: list[str] = []
    for ent in doc.ents:
        if ent.label_ not in ENTITY_LABELS:
            continue
        token = sanitize_token(ent.text)
        if not token or token in seen:
            continue
        seen.add(token)
        entities.append(token)
    return entities[:MAX_ENTITIES_IN_NAME]


def extract_topics(doc: spacy.tokens.Doc) -> list[str]:
    noun_counts: Counter[str] = Counter()
    for token in doc:
        if (
            token.pos_ == "NOUN"
            and not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.text) > 2
        ):
            sanitized = sanitize_token(token.lemma_)
            if sanitized:
                noun_counts[sanitized] += 1
    return [word for word, _ in noun_counts.most_common(MAX_TOPICS_IN_NAME)]


def build_filename(page_number: int, entities: list[str], topics: list[str]) -> str:
    prefix = f"page_{page_number:03d}"
    parts = entities + topics
    if not parts:
        return f"{prefix}_no_text.txt"
    suffix = "_".join(parts)
    base = f"{prefix}_{suffix}"
    max_base_len = MAX_FILENAME_LENGTH - len(".txt")
    if len(base) > max_base_len:
        base = base[:max_base_len].rstrip("_")
    return f"{base}.txt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.isdir(INPUT_DIR):
        print(f"ERROR: Directory not found: '{INPUT_DIR}'")
        sys.exit(1)

    # Gather existing page files sorted by page number
    # Match both original (page_001.txt) and already-renamed (page_001_keywords.txt)
    page_re = re.compile(r"^page_(\d+)(?:_.+)?\.txt$")
    files = []
    for name in sorted(os.listdir(INPUT_DIR)):
        m = page_re.match(name)
        if m:
            files.append((int(m.group(1)), name))

    if not files:
        print("No page_NNN.txt files found.")
        sys.exit(1)

    print(f"Found {len(files)} page files in {INPUT_DIR}")
    print(f"Loading spaCy model '{SPACY_MODEL}' ...")
    nlp = ensure_spacy_model(SPACY_MODEL)

    renamed = 0
    for page_number, old_name in files:
        old_path = os.path.join(INPUT_DIR, old_name)
        with open(old_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Strip repeating header/footer boilerplate before NLP
        text = _BOILERPLATE_RE.sub("", text).strip()

        if not text:
            new_name = build_filename(page_number, [], [])
        else:
            doc = nlp(text)
            entities_found = extract_entities(doc)
            topics_found = extract_topics(doc)
            new_name = build_filename(page_number, entities_found, topics_found)

        new_path = os.path.join(INPUT_DIR, new_name)

        if old_name != new_name:
            os.rename(old_path, new_path)
            renamed += 1

        print(f"  [{page_number:3d}/{len(files)}] {old_name} -> {new_name}")

    print()
    print("=" * 60)
    print(f"  Total files: {len(files)}")
    print(f"  Renamed:     {renamed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
