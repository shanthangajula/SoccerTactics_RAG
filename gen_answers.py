#!/usr/bin/env python3
"""
Phase 1 of 2 — generate + cache answers from the RAG pipeline.

Runs in your EXISTING pipeline venv (langchain 1.x + langgraph). Imports your
graph; does NOT import ragas. Writes one record per question to
evals/generations_{PIPELINE}.jsonl so Phase 2 (score_ragas.py) can score it in
an isolated environment without re-running — or re-paying for — the pipeline.

Usage:
    python gen_answers.py            # uses PIPELINE below; flip to compare
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PIPELINE = "agentic"                              # "agentic" (CRAG) or "naive"
EVAL_SET = Path("evals/eval_set.jsonl")
OUT      = Path(f"evals/generations_{PIPELINE}.jsonl")


def run_rag(question: str) -> tuple[str, list[str]]:
    """Return (answer, retrieved_context_texts) from the chosen pipeline."""
    if PIPELINE == "agentic":
        from agentic_rag import ask                # ask(q) -> full GraphState dict
        final = ask(question)
        return final["answer"], [d.page_content for d in final["documents"]]
    if PIPELINE == "naive":
        from rag import ask                         # ask(q) -> (answer, docs)
        answer, docs = ask(question)
        return answer, [d.page_content for d in docs]
    raise ValueError(f"Unknown PIPELINE: {PIPELINE!r}")


def main() -> None:
    rows = [json.loads(l) for l in EVAL_SET.read_text().splitlines() if l.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for i, r in enumerate(rows, 1):
            answer, contexts = run_rag(r["question"])
            rec = {
                "id": r["id"],
                "category": r["category"],
                "answerable": r.get("answerable", True),
                "question": r["question"],
                "answer": answer,
                "contexts": contexts,
            }
            if "ground_truth" in r:                 # passes through to enable recall
                rec["ground_truth"] = r["ground_truth"]
            f.write(json.dumps(rec) + "\n")
            print(f"[{i}/{len(rows)}] {r['id']}: {len(contexts)} chunks")
    print(f"Wrote {OUT}  ({PIPELINE} pipeline)")


if __name__ == "__main__":
    main()
