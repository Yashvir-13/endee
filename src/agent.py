"""
Agentic RAG — multi-step reasoning with iterative retrieval.

The agent assesses whether retrieved context is sufficient to answer
the question. If not, it reformulates the query and retrieves again.
Max 3 iterations to stay responsive.
"""

import ollama
from src.embeddings import SentenceTransformerEmbedder
from src.endee_client import EndeeClient, INDEX_NAME
from src.sparse import TFIDFSparseEncoder, DEFAULT_VECTORIZER_PATH
from src.reranker import rerank_passages


ASSESS_PROMPT = """You are a retrieval quality assessor. Given a question and retrieved passages, determine if the passages contain enough information to answer the question.

QUESTION: {question}

RETRIEVED PASSAGES:
{context}

Respond with ONLY a JSON object:
{{"sufficient": true/false, "reason": "why", "refined_query": "better search query if not sufficient"}}"""

ANSWER_PROMPT = """You are a scholarly assistant specializing in the philosophy of Arthur Schopenhauer.

RULES:
1. Answer ONLY based on the provided context passages below.
2. If the answer is NOT found in the context, say: "I cannot answer this based on the available passages."
3. Do NOT invent or hallucinate philosophical content.
4. Cite which passage(s) support your answer using [Source: filename, Chunk N] format.
5. Be precise and philosophical in tone.

CONTEXT PASSAGES:
{context}

QUESTION: {question}

Provide a grounded answer citing the relevant passages above."""


def _format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("meta", {})
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        text = meta.get("text", "")
        parts.append(
            f"--- Passage {i} [Source: {source}, Chunk {chunk_idx}] ---\n{text}"
        )
    return "\n\n".join(parts)


def run_agent(
    question: str,
    endee_url: str = "http://localhost:8080/api/v1",
    top_k: int = 5,
    model: str = "llama3",
    max_iterations: int = 3,
) -> None:
    """
    Agentic RAG: iterative retrieval with quality assessment.

    Loop:
      1. Retrieve passages for the current query
      2. Re-rank them
      3. Ask LLM if context is sufficient
      4. If not, refine query and loop (up to max_iterations)
      5. Generate final answer with best available context
    """
    import json

    print("=" * 60)
    print("  PHILOSOPHER RAG — Agentic Mode")
    print("=" * 60)

    # Initialize components
    embedder = SentenceTransformerEmbedder()
    client = EndeeClient(base_url=endee_url)

    # Try loading sparse encoder
    sparse_encoder = None
    try:
        sparse_encoder = TFIDFSparseEncoder.load(DEFAULT_VECTORIZER_PATH)
    except Exception:
        pass

    current_query = question
    all_results = []
    seen_ids = set()

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'─' * 40}")
        print(f"  Iteration {iteration}/{max_iterations}")
        print(f"  Query: \"{current_query}\"")
        print(f"{'─' * 40}")

        # 1. Embed and search
        query_vector = embedder.embed_query(current_query)

        search_kwargs = {
            "index_name": INDEX_NAME,
            "query_vector": query_vector,
            "top_k": top_k,
        }

        if sparse_encoder:
            s_indices, s_values = sparse_encoder.encode(current_query)
            search_kwargs["sparse_indices"] = s_indices
            search_kwargs["sparse_values"] = s_values

        results = client.search(**search_kwargs)

        # 2. Re-rank
        print(f"  Re-ranking {len(results)} passages...")
        results = rerank_passages(current_query, results, model=model, top_n=top_k)

        # 3. Merge with previous results (dedup by ID)
        for r in results:
            if r["id"] not in seen_ids:
                all_results.append(r)
                seen_ids.add(r["id"])

        # 4. Assess sufficiency
        context = _format_context(all_results[:top_k])
        print(f"  Assessing retrieval quality...")

        try:
            assess_response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": ASSESS_PROMPT.format(
                        question=question, context=context
                    ),
                }],
                options={"temperature": 0.0, "num_predict": 200},
            )

            assessment = json.loads(assess_response["message"]["content"].strip())
            sufficient = assessment.get("sufficient", True)
            reason = assessment.get("reason", "")
            refined_query = assessment.get("refined_query", current_query)

            print(f"  Sufficient: {sufficient}")
            print(f"  Reason: {reason}")

            if sufficient or iteration == max_iterations:
                break

            # 5. Refine query for next iteration
            current_query = refined_query
            print(f"  Refined query: \"{refined_query}\"")

        except (json.JSONDecodeError, Exception) as e:
            print(f"  Assessment failed ({e}), proceeding with current context.")
            break

    # 6. Generate final answer
    print(f"\n  Generating final answer...")
    context = _format_context(all_results[:top_k])

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": ANSWER_PROMPT.format(
                context=context, question=question
            )},
        ],
    )

    answer = response["message"]["content"]

    # Print results
    print(f"\n{'─' * 60}")
    print(f"ANSWER (after {iteration} iteration(s)):")
    print(f"{'─' * 60}")
    print(answer)

    print(f"\n{'─' * 60}")
    print("RETRIEVED PASSAGES:")
    print(f"{'─' * 60}")
    for i, r in enumerate(all_results[:top_k], 1):
        meta = r.get("meta", {})
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        text = meta.get("text", "")
        rerank_score = r.get("rerank_score", "n/a")
        preview = text[:250] + "..." if len(text) > 250 else text
        print(f"\n  [{i}] Source: {source}, Chunk {chunk_idx} "
              f"(rerank: {rerank_score}/10)")
        print(f"      {preview}")
    print()
