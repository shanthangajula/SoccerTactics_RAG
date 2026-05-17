"""
Load every .txt under data/raw, chunk it, embed with OpenAI text-embedding-3-small,
persist to ./chroma_db under the collection "soccer_tactics".

Usage:
    python ingestion/index.py            # build the index
    python ingestion/index.py --query "what is gegenpressing?"  # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION = "soccer_tactics"
EMBEDDING_MODEL = "text-embedding-3-small"

# 500 tokens ≈ ~2000 characters. RecursiveCharacterTextSplitter measures characters
# unless given a token-based length function; using a tiktoken-backed splitter keeps
# the chunk size literally at 500 tokens like the spec.
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


def load_documents() -> list:
    if not RAW_DIR.exists():
        sys.exit(f"Missing {RAW_DIR}. Run ingestion/scraper.py first.")
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files in {RAW_DIR}. Run ingestion/scraper.py first.")

    docs = []
    for fp in files:
        loader = TextLoader(str(fp), encoding="utf-8")
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_file"] = fp.name
            d.metadata["topic"] = fp.stem
        docs.extend(loaded)
    print(f"Loaded {len(docs)} documents from {len(files)} files.")
    return docs


def chunk(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks "
          f"({CHUNK_TOKENS}-token chunks, {CHUNK_OVERLAP_TOKENS}-token overlap).")
    return chunks


def build_index() -> Chroma:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Add it to .env.")

    docs = load_documents()
    chunks = chunk(docs)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"Embedding with {EMBEDDING_MODEL} and writing to {CHROMA_DIR} ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Success. Indexed {len(chunks)} chunks into collection '{COLLECTION}'.")
    return vectordb


def load_index() -> Chroma:
    if not CHROMA_DIR.exists():
        sys.exit(f"No index at {CHROMA_DIR}. Run `python ingestion/index.py` first.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def smoke_test(query: str) -> None:
    db = load_index()
    print(f"\nQuery: {query!r}\nTop 3 chunks:\n" + "-" * 60)
    for i, (doc, score) in enumerate(db.similarity_search_with_score(query, k=3), 1):
        topic = doc.metadata.get("topic", "?")
        preview = doc.page_content.replace("\n", " ")[:240]
        print(f"[{i}] score={score:.4f}  topic={topic}")
        print(f"    {preview}...\n")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Skip indexing; just run a similarity query")
    args = parser.parse_args()

    if args.query:
        smoke_test(args.query)
    else:
        build_index()
        # Default smoke test once built.
        smoke_test("What is gegenpressing?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
