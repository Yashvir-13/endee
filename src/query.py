"""
Query pipeline with advanced RAG features.

Supports:
  - Hybrid search (dense + sparse)
  - HyDE (Hypothetical Document Embeddings)
  - Query expansion (multi-query)
  - LLM-based re-ranking
  - Metadata filtering by volume
  - Streaming LLM output
  - Hierarchical context (parent chunks)
"""

import json
import ollama
from src.embeddings import SentenceTransformerEmbedder
from src.endee_client import EndeeClient, INDEX_NAME
from src.sparse import TFIDFSparseEncoder, DEFAULT_VECTORIZER_PATH
from src.reranker import rerank_passages


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a scholarly assistant specializing in the philosophy of Arthur Schopenhauer.

RULES:
1. Answer ONLY based on the provided context passages below.
2. If the answer is NOT found in the context, say: "I cannot answer this based on the available passages."
3. Do NOT invent or hallucinate philosophical content.
4. Cite which passage(s) support your answer using [Source: filename, Chunk N] format.
5. Be precise and philosophical in tone."""

USER_PROMPT_TEMPLATE = """CONTEXT PASSAGES:
{context}

QUESTION: {question}

Provide a grounded answer citing the relevant passages above."""

HYDE_PROMPT = """You are a philosophy scholar. Write a short passage (2-3 sentences) that would appear in Schopenhauer's writings and directly answers this question:

{question}

Write ONLY the hypothetical passage, no preamble."""

EXPAND_PROMPT = """Generate 3 alternative phrasings of this philosophical question for better search retrieval. Each should capture a different aspect or use different terminology.

Original: {question}

Respond with ONLY a JSON list of 3 strings: ["phrasing1", "phrasing2", "phrasing3"]"""


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------
def _format_context(results: list[dict], use_parent: bool = True) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("meta", {})
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        score = r.get("score", 0.0)

        # Use parent text for richer context if available
        if use_parent and meta.get("parent_text"):
            text = meta["parent_text"]
        else:
            text = meta.get("text", "")

        rerank_info = ""
        if "rerank_score" in r:
            rerank_info = f" | rerank: {r['rerank_score']}/10"

        parts.append(
            f"--- Passage {i} [Source: {source}, Chunk {chunk_idx}] "
            f"(similarity: {score:.4f}{rerank_info}) ---\n{text}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Advanced retrieval features
# ---------------------------------------------------------------------------
def _generate_hyde_passage(question: str, model: str) -> str:
    """Generate a hypothetical document for HyDE."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
        options={"temperature": 0.7, "num_predict": 200},
    )
    return response["message"]["content"].strip()


def _expand_query(question: str, model: str) -> list[str]:
    """Generate alternative query phrasings."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": EXPAND_PROMPT.format(question=question),
            }],
            options={"temperature": 0.7, "num_predict": 300},
        )
        phrasings = json.loads(response["message"]["content"].strip())
        if isinstance(phrasings, list):
            return phrasings[:3]
    except (json.JSONDecodeError, Exception):
        pass
    return []


def _build_filter(filter_source: str | None) -> list[dict] | None:
    """Build Endee filter from CLI --filter flag."""
    if not filter_source:
        return None

    vol_map = {
        "vol1": "vol1", "1": "vol1", "volume1": "vol1",
        "vol2": "vol2", "2": "vol2", "volume2": "vol2",
        "vol3": "vol3", "3": "vol3", "volume3": "vol3",
    }
    vol = vol_map.get(filter_source.lower())
    if vol:
        return [{"volume": {"$eq": vol}}]
    return None


# ---------------------------------------------------------------------------
# Main query pipeline
# ---------------------------------------------------------------------------
def run_query(
    question: str,
    endee_url: str = "http://localhost:8080/api/v1",
    top_k: int = 5,
    model: str = "llama3",
    enable_hyde: bool = False,
    enable_expand: bool = False,
    enable_rerank: bool = False,
    enable_streaming: bool = False,
    filter_source: str | None = None,
    use_parent_context: bool = True,
) -> None:
    """Execute the query pipeline with optional advanced features."""

    # Collect active features for display
    features = []
    if enable_hyde:
        features.append("HyDE")
    if enable_expand:
        features.append("Query Expansion")
    if enable_rerank:
        features.append("Re-ranking")
    if filter_source:
        features.append(f"Filter: {filter_source}")
    if enable_streaming:
        features.append("Streaming")

    print("=" * 60)
    print("  PHILOSOPHER RAG — Query Pipeline")
    if features:
        print(f"  Features: {', '.join(features)}")
    print("=" * 60)

    # Initialize components
    embedder = SentenceTransformerEmbedder()
    client = EndeeClient(base_url=endee_url)

    # Try loading sparse encoder for hybrid search
    sparse_encoder = None
    try:
        sparse_encoder = TFIDFSparseEncoder.load(DEFAULT_VECTORIZER_PATH)
    except Exception:
        pass

    # Build filter
    search_filter = _build_filter(filter_source)

    # ---- Step 1: HyDE (optional) ----
    search_text = question
    if enable_hyde:
        print(f"\n[HyDE] Generating hypothetical passage...")
        hyde_passage = _generate_hyde_passage(question, model)
        print(f"  → \"{hyde_passage[:100]}...\"")
        search_text = hyde_passage  # embed the hypothetical passage instead

    # ---- Step 2: Embed query ----
    print(f"\n[1/3] Embedding {'HyDE passage' if enable_hyde else 'question'}...")
    query_vector = embedder.embed_query(search_text)

    # ---- Step 3: Query expansion (optional) ----
    all_results = []
    seen_ids = set()

    if enable_expand:
        print(f"[Expand] Generating alternative queries...")
        alt_queries = _expand_query(question, model)
        for q in alt_queries:
            print(f"  → \"{q[:80]}\"")

        # Search with all queries and merge results
        queries_to_search = [search_text] + alt_queries
        for qi, qt in enumerate(queries_to_search):
            qv = query_vector if qi == 0 else embedder.embed_query(qt)

            kwargs = {"index_name": INDEX_NAME, "query_vector": qv, "top_k": top_k}
            if sparse_encoder:
                si, sv = sparse_encoder.encode(qt)
                kwargs["sparse_indices"] = si
                kwargs["sparse_values"] = sv
            if search_filter:
                kwargs["filter"] = search_filter

            results = client.search(**kwargs)
            for r in results:
                if r["id"] not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r["id"])
    else:
        # Single query search
        print(f"[2/3] Searching Endee for top-{top_k} passages...")
        kwargs = {
            "index_name": INDEX_NAME,
            "query_vector": query_vector,
            "top_k": top_k,
        }
        if sparse_encoder:
            si, sv = sparse_encoder.encode(question)
            kwargs["sparse_indices"] = si
            kwargs["sparse_values"] = sv
        if search_filter:
            kwargs["filter"] = search_filter

        all_results = client.search(**kwargs)

    if not all_results:
        print("\n  No relevant passages found. Run 'python run.py ingest' first.")
        return

    # ---- Step 4: Re-rank (optional) ----
    if enable_rerank:
        print(f"  Re-ranking {len(all_results)} passages with {model}...")
        all_results = rerank_passages(question, all_results, model=model, top_n=top_k)
    else:
        # Sort by score and limit to top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:top_k]

    # ---- Step 5: Generate answer ----
    context = _format_context(all_results, use_parent=use_parent_context)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    if enable_streaming:
        print(f"[3/3] Generating answer (streaming)...\n")
        print("-" * 60)
        print("ANSWER:")
        print("-" * 60)

        stream = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
        print("\n")
    else:
        print(f"[3/3] Generating answer with {model}...\n")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response["message"]["content"]
        print("-" * 60)
        print("ANSWER:")
        print("-" * 60)
        print(answer)
        print()

    # Print retrieved passages
    print("-" * 60)
    print("RETRIEVED PASSAGES:")
    print("-" * 60)
    for i, r in enumerate(all_results, 1):
        meta = r.get("meta", {})
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        text = meta.get("text", "")
        score = r.get("score", 0.0)
        rerank = f", rerank: {r['rerank_score']}/10" if "rerank_score" in r else ""
        preview = text[:250] + "..." if len(text) > 250 else text
        print(f"\n  [{i}] Source: {source}, Chunk {chunk_idx} "
              f"(score: {score:.4f}{rerank})")
        print(f"      {preview}")
    print()

    return all_results  # useful for chat mode


# ---------------------------------------------------------------------------
# Conversation mode
# ---------------------------------------------------------------------------
def run_chat(
    endee_url: str = "http://localhost:8080/api/v1",
    top_k: int = 5,
    model: str = "llama3",
) -> None:
    """Interactive conversation mode with memory."""

    print("=" * 60)
    print("  PHILOSOPHER RAG — Conversation Mode")
    print("  Type 'quit' or 'exit' to end. Type 'clear' to reset.")
    print("=" * 60)

    embedder = SentenceTransformerEmbedder()
    client = EndeeClient(base_url=endee_url)

    sparse_encoder = None
    try:
        sparse_encoder = TFIDFSparseEncoder.load(DEFAULT_VECTORIZER_PATH)
    except Exception:
        pass

    conversation_history = []

    while True:
        try:
            question = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("  Goodbye!")
            break
        if question.lower() == "clear":
            conversation_history.clear()
            print("  Conversation cleared.")
            continue

        # Embed and search
        query_vector = embedder.embed_query(question)
        kwargs = {
            "index_name": INDEX_NAME,
            "query_vector": query_vector,
            "top_k": top_k,
        }
        if sparse_encoder:
            si, sv = sparse_encoder.encode(question)
            kwargs["sparse_indices"] = si
            kwargs["sparse_values"] = sv

        results = client.search(**kwargs)
        context = _format_context(results, use_parent=True)

        # Build conversation with memory
        conversation_history.append({"role": "user", "content": f"""CONTEXT PASSAGES:
{context}

QUESTION: {question}

Answer based on the passages above, citing sources."""})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Keep last 6 messages (3 turns) for context
        messages.extend(conversation_history[-6:])

        # Stream the response
        print("\n  Assistant: ", end="", flush=True)
        full_response = ""
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
            full_response += content
        print()

        conversation_history.append({"role": "assistant", "content": full_response})

        # Show sources briefly
        print(f"\n  📚 Sources: ", end="")
        sources = set()
        for r in results[:3]:
            meta = r.get("meta", {})
            sources.add(f"{meta.get('source', '?')}:Chunk{meta.get('chunk_index', '?')}")
        print(", ".join(sources))
