"""
LLM-based re-ranking via Ollama.

Instead of downloading a separate cross-encoder model, we use the
already-available Llama3 to score passage relevance. This keeps
resource costs zero for evaluators.
"""

import json
import ollama


RERANK_PROMPT = """You are a relevance scoring system. Score how relevant the passage is to the question.

QUESTION: {question}

PASSAGE: {passage}

Score from 1-10 where:
1 = completely irrelevant
5 = somewhat relevant
10 = directly answers the question

Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""


def rerank_passages(
    question: str,
    results: list[dict],
    model: str = "llama3",
    top_n: int = 5,
) -> list[dict]:
    """
    Re-rank retrieved passages using LLM-based relevance scoring.

    Args:
        question: The user's question
        results: List of search results (each with 'meta', 'score', 'id')
        model: Ollama model name
        top_n: Number of top results to return after re-ranking

    Returns:
        Re-ranked results with added 'rerank_score' and 'rerank_reason' fields
    """
    scored = []

    for r in results:
        text = r.get("meta", {}).get("text", "")
        if not text:
            scored.append({**r, "rerank_score": 0, "rerank_reason": "no text"})
            continue

        # Truncate long passages to avoid overwhelming the LLM
        passage = text[:800]

        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": RERANK_PROMPT.format(
                        question=question, passage=passage
                    ),
                }],
                options={"temperature": 0.0, "num_predict": 100},
            )

            answer = response["message"]["content"].strip()
            # Parse the JSON score
            parsed = json.loads(answer)
            score = int(parsed.get("score", 5))
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract a number from the response
            try:
                score = int("".join(c for c in answer if c.isdigit())[:2])
                score = min(max(score, 1), 10)
                reason = "parsed from raw output"
            except Exception:
                score = 5
                reason = "scoring failed, using default"

        scored.append({
            **r,
            "rerank_score": score,
            "rerank_reason": reason,
        })

    # Sort by LLM score (descending), break ties with original score
    scored.sort(key=lambda x: (x["rerank_score"], x["score"]), reverse=True)
    return scored[:top_n]
