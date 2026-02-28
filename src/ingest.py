"""
Ingestion pipeline: load, chunk (hierarchical), embed, sparse-encode, store in Endee.

Usage:
    python run.py ingest [--recreate]
"""

import time
from src.embeddings import SentenceTransformerEmbedder
from src.endee_client import EndeeClient, INDEX_NAME
from src.sparse import TFIDFSparseEncoder, DEFAULT_VECTORIZER_PATH
from src.utils import load_documents, hierarchical_chunk_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCUMENTS_DIR = "Documents"
PARENT_CHUNK_SIZE = 3000   # ~1000 tokens — context for LLM
CHILD_CHUNK_SIZE = 800     # ~250 tokens — precise retrieval unit
CHILD_OVERLAP = 100


def run_ingestion(
    endee_url: str = "http://localhost:8080/api/v1",
    recreate_index: bool = False,
) -> None:
    """Execute the full ingestion pipeline with hierarchical chunking and sparse vectors."""

    print("=" * 60)
    print("  PHILOSOPHER RAG — Ingestion Pipeline")
    print("  Features: Hierarchical chunking, Hybrid search (TF-IDF)")
    print("=" * 60)

    # 1. Load documents
    print("\n[1/5] Loading documents...")
    docs = load_documents(DOCUMENTS_DIR)
    print(f"  Loaded {len(docs)} document(s).\n")

    # 2. Hierarchical chunking
    print("[2/5] Chunking documents (hierarchical: parent→child)...")
    all_chunks = []
    for filename, content in docs:
        chunks = hierarchical_chunk_text(
            content,
            source=filename,
            parent_size=PARENT_CHUNK_SIZE,
            child_size=CHILD_CHUNK_SIZE,
            child_overlap=CHILD_OVERLAP,
        )
        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} child chunks")

    print(f"  Total child chunks: {len(all_chunks)}\n")

    # 3. Dense embeddings
    print("[3/5] Embedding chunks (this may take a moment)...")
    embedder = SentenceTransformerEmbedder()
    texts = [c.text for c in all_chunks]
    t0 = time.time()
    vectors = embedder.embed_passages(texts)
    elapsed = time.time() - t0
    print(f"  Embedded {len(vectors)} chunks in {elapsed:.1f}s "
          f"(dim={embedder.dimension()})\n")

    # 4. Sparse encoding (TF-IDF)
    print("[4/5] Building TF-IDF sparse vectors...")
    sparse_encoder = TFIDFSparseEncoder()
    sparse_encoder.fit(texts)
    sparse_encoder.save(DEFAULT_VECTORIZER_PATH)
    sparse_vectors = sparse_encoder.encode_batch(texts)
    print(f"  Generated sparse vectors (sparse_dim={sparse_encoder.dim})\n")

    # 5. Store in Endee
    print("[5/5] Storing vectors in Endee (hybrid index)...")
    client = EndeeClient(base_url=endee_url)

    if recreate_index:
        client.delete_index(INDEX_NAME)

    client.create_index(
        name=INDEX_NAME,
        dimension=embedder.dimension(),
        sparse_dim=sparse_encoder.dim,
    )

    # Prepare IDs, metadata, and filters
    ids = [f"{c.source}__chunk_{c.chunk_index}" for c in all_chunks]
    metadatas = [
        {
            "text": c.text,
            "source": c.source,
            "chunk_index": c.chunk_index,
            "parent_text": c.parent_text if c.parent_text else c.text,
        }
        for c in all_chunks
    ]

    # Volume label for metadata filtering (e.g., "vol1", "vol2", "vol3")
    filters = []
    for c in all_chunks:
        vol = "unknown"
        if "vol1" in c.source.lower():
            vol = "vol1"
        elif "vol2" in c.source.lower():
            vol = "vol2"
        elif "vol3" in c.source.lower():
            vol = "vol3"
        filters.append({"volume": vol})

    inserted = client.upsert_vectors(
        index_name=INDEX_NAME,
        ids=ids,
        vectors=vectors,
        metadatas=metadatas,
        sparse_vectors=sparse_vectors,
        filters=filters,
    )

    print(f"\n{'=' * 60}")
    print(f"  Ingestion complete!")
    print(f"  Documents:    {len(docs)}")
    print(f"  Chunks:       {len(all_chunks)} (hierarchical)")
    print(f"  Vectors:      {inserted} (dense + sparse)")
    print(f"  Index:        {INDEX_NAME}")
    print(f"  Sparse dim:   {sparse_encoder.dim}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_ingestion()
