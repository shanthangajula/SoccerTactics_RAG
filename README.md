# soccer-tactics-rag

Compound AI workflow combining RAG, planned agentic orchestration, and evaluation harness over soccer tactical analysis.

This repo currently contains a **naive RAG baseline** over 20 Wikipedia articles covering tactics, formations, and player roles. Agentic orchestration and evals come next.

## Stack

- Embeddings: OpenAI `text-embedding-3-small`
- Vector store: Chroma (local, persisted to `./chroma_db`)
- Generator: Claude Sonnet via `langchain-anthropic`
- Chunking: 500 tokens, 50-token overlap, `RecursiveCharacterTextSplitter`

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then put real keys in .env
```

## Build the index

```bash
python ingestion/scraper.py        # writes data/raw/*.txt (~20 files)
python ingestion/index.py          # builds chroma_db/, runs a smoke query
```

## Query

```bash
python rag.py "What is gegenpressing?"
python rag.py "How does a false nine differ from a target man?" --show-sources
```

Or from Python:

```python
from rag import ask
answer, sources = ask("Compare 4-3-3 to 4-2-3-1")
```

## Layout

```
soccer-tactics-rag/
  urls.txt                 # curated Wikipedia URLs
  ingestion/
    scraper.py             # fetch + clean Wikipedia HTML -> data/raw/*.txt
    index.py               # chunk + embed + persist to chroma_db/
  rag.py                   # ask() + CLI
  data/raw/                # gitignored, populated by scraper
  chroma_db/               # gitignored, populated by index
  .env                     # gitignored, see .env.example
```

## URL choices

A few items from the original brief were merged or substituted because Wikipedia handles them as redirects or sections of broader articles rather than standalone pages:

- "Gegenpressing" redirects to **Counter-press**, so the list keeps Counter-press once and uses **Catenaccio** for the freed slot.
- "Possession football" has no standalone article; replaced with **Formation (association football)**, which gives broad tactical context.
- "Inverted fullback", "deep-lying playmaker", "holding midfielder", "false 10", and "target man" don't have dedicated articles. They're covered as sections of the broader position pages: **Defender**, **Playmaker**, **Midfielder**, **Forward**, **Goalkeeper**.

If you'd rather have stricter 1:1 mapping, swap entries in `urls.txt` and re-run the scraper.
