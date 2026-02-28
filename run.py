#!/usr/bin/env python3
"""
Philosopher RAG — CLI entry point.

Usage:
    python run.py ingest                   # Index documents into Endee
    python run.py query "question"         # Basic query
    python run.py query "question" --all   # All advanced features
    python run.py chat                     # Conversation mode
    python run.py agent "question"         # Agentic multi-step RAG
    python run.py evaluate                 # Run evaluation suite
"""

import argparse
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Philosopher RAG: Grounded QA over Schopenhauer using Endee",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py ingest
  python run.py query "Why does Schopenhauer believe suffering is fundamental?"
  python run.py query "What is the role of art?" --hyde --rerank --stream
  python run.py query "What is the will?" --all
  python run.py query "Discuss volume 1's view on art" --filter vol1
  python run.py chat
  python run.py agent "What is the relationship between will and suffering?"
  python run.py evaluate
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- ingest ----
    ingest_parser = subparsers.add_parser(
        "ingest", help="Index documents into Endee (hierarchical + hybrid)"
    )
    ingest_parser.add_argument(
        "--recreate", action="store_true",
        help="Delete and recreate the index before ingesting"
    )

    # ---- query ----
    query_parser = subparsers.add_parser(
        "query", help="Ask a philosophical question"
    )
    query_parser.add_argument("question", type=str)
    query_parser.add_argument("--top_k", type=int, default=5)
    query_parser.add_argument(
        "--model", type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3"),
    )
    query_parser.add_argument(
        "--hyde", action="store_true",
        help="Enable HyDE (Hypothetical Document Embeddings)"
    )
    query_parser.add_argument(
        "--expand", action="store_true",
        help="Enable query expansion (multi-query)"
    )
    query_parser.add_argument(
        "--rerank", action="store_true",
        help="Enable LLM-based re-ranking"
    )
    query_parser.add_argument(
        "--stream", action="store_true",
        help="Enable streaming LLM output"
    )
    query_parser.add_argument(
        "--filter", type=str, default=None,
        help="Filter by volume: vol1, vol2, vol3"
    )
    query_parser.add_argument(
        "--all", action="store_true",
        help="Enable all advanced features (HyDE + expand + rerank + stream)"
    )

    # ---- chat ----
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive conversation mode with memory"
    )
    chat_parser.add_argument("--top_k", type=int, default=5)
    chat_parser.add_argument(
        "--model", type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3"),
    )

    # ---- agent ----
    agent_parser = subparsers.add_parser(
        "agent", help="Agentic multi-step RAG with iterative retrieval"
    )
    agent_parser.add_argument("question", type=str)
    agent_parser.add_argument("--top_k", type=int, default=5)
    agent_parser.add_argument(
        "--model", type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3"),
    )
    agent_parser.add_argument(
        "--max_iterations", type=int, default=3,
        help="Max retrieval iterations (default: 3)"
    )

    # ---- evaluate ----
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run the evaluation suite"
    )
    eval_parser.add_argument("--top_k", type=int, default=5)

    # ---- Parse and dispatch ----
    args = parser.parse_args()
    endee_url = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")

    if args.command == "ingest":
        from src.ingest import run_ingestion
        run_ingestion(endee_url=endee_url, recreate_index=args.recreate)

    elif args.command == "query":
        from src.query import run_query

        # --all enables everything
        if args.all:
            args.hyde = True
            args.expand = True
            args.rerank = True
            args.stream = True

        run_query(
            question=args.question,
            endee_url=endee_url,
            top_k=args.top_k,
            model=args.model,
            enable_hyde=args.hyde,
            enable_expand=args.expand,
            enable_rerank=args.rerank,
            enable_streaming=args.stream,
            filter_source=args.filter,
        )

    elif args.command == "chat":
        from src.query import run_chat
        run_chat(
            endee_url=endee_url,
            top_k=args.top_k,
            model=args.model,
        )

    elif args.command == "agent":
        from src.agent import run_agent
        run_agent(
            question=args.question,
            endee_url=endee_url,
            top_k=args.top_k,
            model=args.model,
            max_iterations=args.max_iterations,
        )

    elif args.command == "evaluate":
        from src.evaluate import run_evaluation
        run_evaluation(endee_url=endee_url, top_k=args.top_k)


if __name__ == "__main__":
    main()
