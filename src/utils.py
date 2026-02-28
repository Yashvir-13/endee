"""
Utility functions for document loading and text chunking.

Supports both flat chunking and hierarchical (parent-child) chunking
for improved RAG retrieval.
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk with provenance metadata."""
    text: str
    source: str          # source filename
    chunk_index: int     # position within the source document
    parent_text: str = ""  # parent context for hierarchical chunking


def load_documents(directory: str) -> list[tuple[str, str]]:
    """
    Load all .txt files from the given directory.
    Returns list of (filename, content) tuples.
    """
    docs = []
    doc_dir = Path(directory)

    if not doc_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    for filepath in sorted(doc_dir.glob("*.txt")):
        content = filepath.read_text(encoding="utf-8", errors="replace")
        docs.append((filepath.name, content))
        print(f"  Loaded: {filepath.name} ({len(content):,} chars)")

    if not docs:
        raise FileNotFoundError(f"No .txt files found in {directory}")

    return docs


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Split text into overlapping chunks (flat mode).

    Uses character-based splitting with paragraph-boundary awareness.
    - chunk_size:  target size in characters (~500 tokens ≈ 1500 chars)
    - overlap:     overlap between consecutive chunks in characters
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph boundary (double newline)
        if end < len(text):
            break_point = text.rfind("\n\n", start + chunk_size // 2, end + overlap)
            if break_point != -1:
                end = break_point

        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunks.append(Chunk(
                text=chunk_text_str,
                source=source,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

        start = max(start + 1, end - overlap)

    return chunks


def hierarchical_chunk_text(
    text: str,
    source: str,
    parent_size: int = 3000,
    child_size: int = 800,
    child_overlap: int = 100,
) -> list[Chunk]:
    """
    Hierarchical (parent-child) chunking.

    Creates large parent chunks for context, then splits each into
    smaller child chunks for precise retrieval. Each child stores
    its parent's full text in metadata for LLM context.

    - parent_size:   large context window (~1000 tokens)
    - child_size:    small retrieval chunk (~250 tokens)
    - child_overlap: overlap between child chunks
    """
    chunks = []
    chunk_index = 0

    # Step 1: Create parent chunks
    parent_start = 0
    while parent_start < len(text):
        parent_end = parent_start + parent_size

        # Try to break at paragraph boundary
        if parent_end < len(text):
            bp = text.rfind("\n\n", parent_start + parent_size // 2, parent_end + 200)
            if bp != -1:
                parent_end = bp

        parent_text = text[parent_start:parent_end].strip()

        if not parent_text:
            parent_start = parent_end
            continue

        # Step 2: Split parent into child chunks
        child_start = 0
        while child_start < len(parent_text):
            child_end = child_start + child_size

            # Paragraph-aware breaking within the child
            if child_end < len(parent_text):
                bp = parent_text.rfind(
                    "\n\n", child_start + child_size // 2, child_end + child_overlap
                )
                if bp != -1:
                    child_end = bp

            child_text = parent_text[child_start:child_end].strip()

            if child_text:
                chunks.append(Chunk(
                    text=child_text,
                    source=source,
                    chunk_index=chunk_index,
                    parent_text=parent_text,
                ))
                chunk_index += 1

            child_start = max(child_start + 1, child_end - child_overlap)

        parent_start = parent_end

    return chunks
