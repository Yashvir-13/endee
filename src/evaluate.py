"""
Evaluation pipeline for the Philosopher RAG system.

Runs a curated set of questions, measures retrieval relevance
(concept hit rate) and latency, and prints a summary report.
"""

import json
import time
from src.embeddings import SentenceTransformerEmbedder
from src.endee_client import EndeeClient, INDEX_NAME
from src.sparse import TFIDFSparseEncoder, DEFAULT_VECTORIZER_PATH


EVAL_QUESTIONS_PATH = "eval_questions.json"


def run_evaluation(
    endee_url: str = "http://localhost:8080/api/v1",
    top_k: int = 5,
) -> None:
    """Run the evaluation suite and print a report."""

    print("=" * 60)
    print("  PHILOSOPHER RAG — Evaluation Pipeline")
    print("=" * 60)

    # Load questions
    with open(EVAL_QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    print(f"\n  Loaded {len(questions)} evaluation questions.\n")

    # Initialize components
    print("  Loading embedder...")
    embedder = SentenceTransformerEmbedder()
    client = EndeeClient(base_url=endee_url)

    sparse_encoder = None
    try:
        sparse_encoder = TFIDFSparseEncoder.load(DEFAULT_VECTORIZER_PATH)
        print("  Loaded sparse encoder for hybrid search.")
    except Exception:
        print("  Sparse encoder not available, using dense-only search.")

    # Run evaluation
    total_concept_hits = 0
    total_concepts = 0
    total_time = 0
    results_log = []

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected = q["expected_concepts"]

        print(f"\n  [{i}/{len(questions)}] {question}")

        # Time the retrieval
        t0 = time.time()
        query_vector = embedder.embed_query(question)

        search_kwargs = {
            "index_name": INDEX_NAME,
            "query_vector": query_vector,
            "top_k": top_k,
        }

        if sparse_encoder:
            s_indices, s_values = sparse_encoder.encode(question)
            search_kwargs["sparse_indices"] = s_indices
            search_kwargs["sparse_values"] = s_values

        results = client.search(**search_kwargs)
        elapsed = time.time() - t0
        total_time += elapsed

        # Check concept coverage in retrieved passages
        retrieved_text = " ".join(
            r.get("meta", {}).get("text", "").lower() for r in results
        )

        hits = 0
        for concept in expected:
            if concept.lower() in retrieved_text:
                hits += 1

        hit_rate = hits / len(expected) if expected else 0
        total_concept_hits += hits
        total_concepts += len(expected)

        status = "✅" if hit_rate >= 0.6 else "⚠️" if hit_rate >= 0.4 else "❌"
        print(f"    {status} Concept hit rate: {hits}/{len(expected)} "
              f"({hit_rate:.0%}) | Time: {elapsed:.2f}s")

        results_log.append({
            "question": question,
            "hit_rate": hit_rate,
            "hits": hits,
            "total": len(expected),
            "time": elapsed,
        })

    # Print summary
    overall_hit_rate = total_concept_hits / total_concepts if total_concepts else 0
    avg_time = total_time / len(questions) if questions else 0
    pass_count = sum(1 for r in results_log if r["hit_rate"] >= 0.6)

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION REPORT")
    print(f"{'=' * 60}")
    print(f"  Questions:              {len(questions)}")
    print(f"  Passing (≥60% hits):    {pass_count}/{len(questions)}")
    print(f"  Overall concept hit:    {total_concept_hits}/{total_concepts} "
          f"({overall_hit_rate:.0%})")
    print(f"  Avg retrieval time:     {avg_time:.2f}s")
    print(f"  Total time:             {total_time:.1f}s")
    print(f"{'=' * 60}")

    # Detailed breakdown
    print(f"\n  DETAILED BREAKDOWN:")
    for r in results_log:
        status = "✅" if r["hit_rate"] >= 0.6 else "⚠️" if r["hit_rate"] >= 0.4 else "❌"
        print(f"    {status} [{r['hits']}/{r['total']}] "
              f"{r['question'][:60]}...")
    print()
