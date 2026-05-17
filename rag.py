"""
Naive RAG over the soccer-tactics Chroma index.

Usage:
    python rag.py "What is gegenpressing?"
    python rag.py "Compare a 4-3-3 to a 4-2-3-1"

Programmatic:
    from rag import ask
    answer, sources = ask("What is a false nine?")
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

REPO_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION = "soccer_tactics"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-6"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a soccer tactics analyst. Answer the user's question using ONLY the "
    "passages provided in <context>. If the passages don't contain the answer, say "
    "you don't know — do not invent details. Be concrete and concise. When useful, "
    "cite the topic in brackets like [tiki-taka]."
)

USER_TEMPLATE = """<context>
{context}
</context>

Question: {question}

Answer:"""


def _check_env() -> None:
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing env var(s): {', '.join(missing)}. Add them to .env.")


def _load_retriever() -> Chroma:
    if not CHROMA_DIR.exists():
        sys.exit(f"No index at {CHROMA_DIR}. Run `python ingestion/index.py` first.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks as a labeled block for the LLM."""
    blocks = []
    for i, d in enumerate(docs, 1):
        topic = d.metadata.get("topic", "unknown")
        blocks.append(f"[{i}] topic={topic}\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def ask(question: str, k: int = TOP_K) -> tuple[str, list[Document]]:
    """Retrieve top-k chunks, ask Claude, return (answer, sources)."""
    _check_env()
    db = _load_retriever()
    docs = db.similarity_search(question, k=k)
    context = _format_context(docs)

    llm = ChatAnthropic(model=CHAT_MODEL, temperature=0, max_tokens=1024)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", USER_TEMPLATE.format(context=context, question=question)),
    ]
    response = llm.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return answer, docs


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask the soccer-tactics RAG.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved chunks")
    args = parser.parse_args()

    answer, sources = ask(args.question, k=args.k)
    print("\n" + "=" * 70)
    print(f"Q: {args.question}")
    print("=" * 70)
    print(answer.strip())
    print("=" * 70)
    print(f"Sources: {', '.join(sorted({d.metadata.get('topic', '?') for d in sources}))}")
    if args.show_sources:
        print("\n--- Retrieved chunks ---")
        for i, d in enumerate(sources, 1):
            print(f"\n[{i}] {d.metadata.get('topic')}")
            print(d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
