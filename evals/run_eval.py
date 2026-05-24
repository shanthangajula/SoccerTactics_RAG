"""
Evaluation harness for the soccer-tactics RAG.

For each ground-truth question we measure three things:

  1. keyword_recall   — fraction of expected keywords present in the answer
                        (cheap, deterministic, catches obvious failures)

  2. topic_hit        — did the retriever surface a chunk from the topic(s)
                        we expected? (isolates retrieval from generation)

  3. faithfulness     — (optional, LLM-as-judge with gpt-4o-mini) does every
                        claim in the answer follow from the retrieved chunks?

The three together separate the two failure modes a RAG can have:
retrieval failure (topic_hit drops) vs generation failure (faithfulness
drops while topic_hit stays high).

Usage:
    python evals/run_eval.py                     # deterministic metrics only
    python evals/run_eval.py --judge             # add LLM-as-judge faithfulness
    python evals/run_eval.py --judge --out evals/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Make the repo root importable so we can use rag.ask()
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rag import ask  # noqa: E402

GROUND_TRUTH = REPO_ROOT / "evals" / "ground_truth.json"
JUDGE_MODEL = "gpt-4o-mini"  # cheap, ~$0.15 / 1M input tokens

JUDGE_SYSTEM = (
    "You are a strict evaluator. You will be given a set of CONTEXT passages "
    "retrieved from a knowledge base, plus an ANSWER produced by a RAG system. "
    "Your job is to score the answer on FAITHFULNESS: how well every factual "
    "claim in the answer is supported by the context. Hallucinations, unsupported "
    "specifics, or invented details should be penalised. If the answer says 'I "
    "don't know' and that is appropriate given the context, score it 1.0. "
    "Reply ONLY with a JSON object of the form: "
    '{"faithfulness": <float 0.0-1.0>, "reason": "<one short sentence>"}'
)

JUDGE_USER_TEMPLATE = """CONTEXT:
{context}

ANSWER:
{answer}

Score the answer's faithfulness to the context."""


# ---------- deterministic scoring ----------

def keyword_recall(answer: str, expected: list[str]) -> float:
    """Fraction of expected keywords that appear in the answer (case-insensitive)."""
    if not expected:
        return 1.0
    text = answer.lower()
    hits = sum(1 for kw in expected if kw.lower() in text)
    return hits / len(expected)


def topic_hit(retrieved_topics: list[str], expected_topics: list[str]) -> float:
    """1.0 if any expected topic appears among retrieved topics, else 0.0.

    We use 'any' rather than 'all' because top-k is bounded and one good chunk
    is usually enough to answer. Switch to all() for a stricter recall measure.
    """
    if not expected_topics:
        return 1.0
    return 1.0 if any(t in retrieved_topics for t in expected_topics) else 0.0


# ---------- LLM-as-judge faithfulness ----------

def llm_judge_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """Use gpt-4o-mini to score how well `answer` is grounded in `context`."""
    from openai import OpenAI  # local import: only needed when --judge is on

    client = OpenAI()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(context=context, answer=answer)},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        score = float(parsed.get("faithfulness", 0.0))
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        score, reason = 0.0, f"parse_error: {raw[:100]}"
    return {"score": max(0.0, min(1.0, score)), "reason": reason}


# ---------- main loop ----------

def run(use_judge: bool, out_path: Path | None) -> int:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")

    ground_truth = json.loads(GROUND_TRUTH.read_text())
    print(f"Loaded {len(ground_truth)} ground-truth questions.")
    print(f"Mode: {'with LLM-as-judge faithfulness' if use_judge else 'deterministic only'}")
    print()

    results: list[dict[str, Any]] = []
    for i, case in enumerate(ground_truth, 1):
        qid = case["id"]
        question = case["question"]
        print(f"[{i:>2}/{len(ground_truth)}] {qid}: {question}")

        t0 = time.time()
        try:
            answer, sources = ask(question)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR calling ask(): {exc}")
            results.append({"id": qid, "error": str(exc)})
            continue
        latency = time.time() - t0

        retrieved_topics = sorted({d.metadata.get("topic", "?") for d in sources})
        kw_score = keyword_recall(answer, case.get("expected_keywords", []))
        topic_score = topic_hit(retrieved_topics, case.get("topic_should_appear", []))

        row: dict[str, Any] = {
            "id": qid,
            "question": question,
            "answer": answer.strip(),
            "retrieved_topics": retrieved_topics,
            "expected_keywords": case.get("expected_keywords", []),
            "expected_topics": case.get("topic_should_appear", []),
            "keyword_recall": round(kw_score, 3),
            "topic_hit": round(topic_score, 3),
            "latency_sec": round(latency, 2),
        }

        if use_judge:
            context_blob = "\n\n---\n\n".join(
                f"[{j}] topic={d.metadata.get('topic')}\n{d.page_content}"
                for j, d in enumerate(sources, 1)
            )
            judged = llm_judge_faithfulness(answer, context_blob)
            row["faithfulness"] = round(judged["score"], 3)
            row["faithfulness_reason"] = judged["reason"]

        results.append(row)

        # Per-question summary line
        flags = f"kw={kw_score:.2f}  topic={topic_score:.0f}"
        if use_judge:
            flags += f"  faith={row['faithfulness']:.2f}"
        flags += f"  ({latency:.1f}s)"
        print(f"    {flags}")
        print(f"    A: {answer.strip()[:140]}{'...' if len(answer) > 140 else ''}")
        print()

    # ---------- aggregate ----------
    scored = [r for r in results if "error" not in r]
    if not scored:
        print("No successful evaluations.")
        return 1

    def avg(key: str) -> float:
        vals = [r[key] for r in scored if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    print("=" * 60)
    print(f"SUMMARY ({len(scored)}/{len(results)} succeeded)")
    print("=" * 60)
    print(f"  keyword_recall  (mean): {avg('keyword_recall'):.3f}")
    print(f"  topic_hit       (mean): {avg('topic_hit'):.3f}")
    if use_judge:
        print(f"  faithfulness    (mean): {avg('faithfulness'):.3f}")
    print(f"  latency_sec     (mean): {avg('latency_sec'):.2f}")
    print("=" * 60)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Detailed results written to {out_path}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--judge", action="store_true",
                   help=f"Use LLM-as-judge ({JUDGE_MODEL}) for faithfulness scoring")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write per-question JSON results")
    args = p.parse_args()
    return run(use_judge=args.judge, out_path=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
