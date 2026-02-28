"""
Endee vector database client wrapper.

Isolates all Endee interactions behind a clean interface.
Supports dense search, hybrid search (dense + sparse), and metadata filtering.
"""

import json
from endee import Endee, Precision


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_ENDEE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "schopenhauer_rag"
SPACE_TYPE = "cosine"
INDEX_PRECISION = Precision.FLOAT32


class EndeeClient:
    """Thin wrapper around the Endee Python SDK with hybrid search support."""

    def __init__(self, base_url: str = DEFAULT_ENDEE_URL):
        self.client = Endee()
        self.client.set_base_url(base_url)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def create_index(
        self, name: str, dimension: int, sparse_dim: int = 0
    ) -> None:
        """Create a vector index. Silently skips if it already exists."""
        try:
            kwargs = {
                "name": name,
                "dimension": dimension,
                "space_type": SPACE_TYPE,
                "precision": INDEX_PRECISION,
            }
            if sparse_dim > 0:
                kwargs["sparse_dim"] = sparse_dim

            self.client.create_index(**kwargs)
            mode = "hybrid" if sparse_dim > 0 else "dense"
            print(f"[Endee] Created {mode} index '{name}' "
                  f"(dim={dimension}, sparse_dim={sparse_dim})")
        except Exception as exc:
            if "already exists" in str(exc).lower():
                print(f"[Endee] Index '{name}' already exists, reusing.")
            else:
                raise

    def delete_index(self, name: str) -> None:
        """Delete the index via REST API (SDK doesn't expose this)."""
        import httpx
        try:
            base = getattr(self.client, 'base_url', DEFAULT_ENDEE_URL)
            url = f"{base}/index/{name}/delete"
            resp = httpx.delete(url, headers={"Authorization": ""}, timeout=30)
            if resp.status_code == 200:
                print(f"[Endee] Deleted index '{name}'")
            else:
                print(f"[Endee] Delete index response: {resp.status_code}")
        except Exception as e:
            print(f"[Endee] Could not delete index '{name}': {e}")

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------
    def upsert_vectors(
        self,
        index_name: str,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
        filters: list[dict] | None = None,
        batch_size: int = 100,
    ) -> int:
        """
        Batch-upsert vectors with metadata, optional sparse vectors, and filters.

        Args:
            ids: Vector IDs
            vectors: Dense vectors
            metadatas: Metadata dicts
            sparse_vectors: Optional list of (indices, values) tuples
            filters: Optional list of filter dicts per vector
            batch_size: Batch size for upsert operations

        Returns:
            Number of vectors inserted
        """
        index = self.client.get_index(name=index_name)
        total = len(ids)
        inserted = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = []
            for i in range(start, end):
                item = {
                    "id": ids[i],
                    "vector": vectors[i],
                    "meta": metadatas[i],
                }
                # Add sparse vectors if available
                if sparse_vectors and i < len(sparse_vectors):
                    s_indices, s_values = sparse_vectors[i]
                    if s_indices:
                        item["sparse_indices"] = s_indices
                        item["sparse_values"] = s_values
                    else:
                        # Hybrid index requires sparse data on every vector;
                        # use a minimal fallback for chunks with empty TF-IDF
                        item["sparse_indices"] = [0]
                        item["sparse_values"] = [1e-7]
                # Add filter if available
                if filters and i < len(filters):
                    item["filter"] = filters[i]

                batch.append(item)

            index.upsert(batch)
            inserted += len(batch)
            print(f"  [Endee] Upserted {inserted}/{total} vectors", end="\r")

        print()
        return inserted

    def search(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int = 5,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        filter: list[dict] | None = None,
    ) -> list[dict]:
        """
        Perform similarity search with optional hybrid and filter support.

        Args:
            index_name: Name of the index
            query_vector: Dense query vector
            top_k: Number of results
            sparse_indices: Optional sparse query indices
            sparse_values: Optional sparse query values
            filter: Optional filter conditions

        Returns:
            List of result dicts with id, score, and meta
        """
        index = self.client.get_index(name=index_name)

        search_kwargs = {
            "vector": query_vector,
            "top_k": top_k,
        }

        if sparse_indices and sparse_values:
            search_kwargs["sparse_indices"] = sparse_indices
            search_kwargs["sparse_values"] = sparse_values

        if filter:
            search_kwargs["filter"] = filter

        raw_results = index.query(**search_kwargs)

        results = []
        for item in raw_results:
            # Handle both dict and object-style results
            if isinstance(item, dict):
                item_id = item.get("id", "")
                score = item.get("similarity", item.get("score", 0.0))
                raw_meta = item.get("meta", {})
            else:
                item_id = getattr(item, "id", "")
                score = getattr(item, "similarity", getattr(item, "score", 0.0))
                raw_meta = getattr(item, "meta", {})

            # Parse meta: could be dict, JSON string, or bytes
            meta = {}
            if isinstance(raw_meta, dict):
                meta = raw_meta
            elif isinstance(raw_meta, str):
                try:
                    meta = json.loads(raw_meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {"raw": raw_meta}
            elif isinstance(raw_meta, bytes):
                try:
                    meta = json.loads(raw_meta.decode("utf-8"))
                except Exception:
                    meta = {"raw": raw_meta.decode("utf-8", errors="replace")}

            results.append({
                "id": item_id,
                "score": score,
                "meta": meta,
            })
        return results
