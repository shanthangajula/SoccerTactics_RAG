#!/usr/bin/env python3
"""
RAGAS baseline for SoccerTactics_RAG.

Runs the eval set through the live CRAG pipeline, scores the *answerable*
questions with RAGAS, scores the *out-of-scope* questions on abstention, and
writes a per-category breakdown to evals/results.md.

Metrics:
  - faithfulness            (no reference needed)
  - answer_relevancy        (no reference needed)
  - context_precision       (LLMContextPrecisionWithoutReference; no reference)
  - context_recall          (LLMContextRecall; ONLY if `ground_truth` is present)

Usage:
    export OPENAI_API_KEY=...
    python ragas_baseline.py

Tested against: ragas>=0.2,<0.3 + langchain-openai. Pin these — RAGAS's metric
API and column names shift across minor versions.
"""

from __future__ import annotations

import json
import math
import datetime as dt
from pathlib import Path
from collections import defaultdict

# Load .env (OPENAI_API_KEY for the judge + embeddings; ANTHROPIC_API_KEY for
# the pipeline's Claude generator) before any client initializes.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper  # if this fails: ragas.embeddings.base
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PIPELINE    = "agentic"          # "agentic" (CRAG) or "naive" — run both to compare
EVAL_SET    = Path("evals/eval_set.jsonl")
RESULTS_MD  = Path(f"evals/results_{PIPELINE}.md")
JUDGE_MODEL = "gpt-4o-mini"      # RAGAS judge (your generator is Claude, set in the pipeline)
EMBED_MODEL = "text-embedding-3-small"

# A correct out-of-scope answer abstains rather than fabricating. These markers
# decide whether the model abstained. Tune to your prompt's refusal wording.
ABSTENTION_MARKERS = [
    "don't have", "do not have", "don't know", "do not know", "i don't know",
    "not in", "knowledge base", "no information", "no relevant",
    "not contain", "does not contain", "doesn't contain",
    "not mention", "does not mention", "doesn't mention",
    "cannot answer", "can't answer", "unable to", "outside the scope",
    "not covered", "not enough information", "insufficient",
    "cannot determine", "can't determine",
]


# --------------------------------------------------------------------------- #
# RAG adapter  --  THE ONE PIECE YOU MUST WIRE TO YOUR PIPELINE
# --------------------------------------------------------------------------- #
def run_rag(question: str) -> tuple[str, list[str]]:
    """Call the RAG pipeline; return (answer, retrieved_contexts).

    retrieved_contexts is the list of raw chunk texts the retriever surfaced for
    THIS question — RAGAS scores context precision/recall against these. The
    agentic path returns the final post-rewrite document set.
    """
    if PIPELINE == "agentic":
        from agentic_rag import ask          # ask(q) -> full GraphState dict
        final = ask(question)
        return final["answer"], [d.page_content for d in final["documents"]]
    if PIPELINE == "naive":
        from rag import ask                   # ask(q) -> (answer, docs)
        answer, docs = ask(question)
        return answer, [d.page_content for d in docs]
    raise ValueError(f"Unknown PIPELINE: {PIPELINE!r}")


# --------------------------------------------------------------------------- #
def load_eval_set(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def nanmean(xs: list[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def is_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in ABSTENTION_MARKERS)


def fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> None:
    rows = load_eval_set(EVAL_SET)
    answerable   = [r for r in rows if r.get("answerable", True)]
    out_of_scope = [r for r in rows if not r.get("answerable", True)]

    # 1) generate answers + contexts from the live pipeline -------------- #
    print(f"Running pipeline on {len(rows)} questions...")
    generated: dict[str, tuple[str, list[str]]] = {}
    for r in rows:
        generated[r["id"]] = run_rag(r["question"])

    # 2) choose metrics based on reference availability ------------------ #
    has_reference = any("ground_truth" in r for r in answerable)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBED_MODEL))

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
    if has_reference:
        metrics.append(LLMContextRecall(llm=evaluator_llm))
        metric_names.append("context_recall")

    # 3) build the RAGAS dataset (answerable only) ----------------------- #
    samples, ids, cats = [], [], []
    for r in answerable:
        ans, ctx = generated[r["id"]]
        kw = dict(user_input=r["question"], response=ans, retrieved_contexts=ctx)
        if "ground_truth" in r:
            kw["reference"] = r["ground_truth"]
        samples.append(SingleTurnSample(**kw))
        ids.append(r["id"])
        cats.append(r["category"])

    print(f"Scoring {len(samples)} answerable questions: {', '.join(metric_names)}")
    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics)
    df = result.to_pandas()
    df.insert(0, "id", ids)
    df.insert(1, "category", cats)

    # column names vary by version — map defensively
    col_map = {}
    for want, cands in {
        "faithfulness":      ["faithfulness"],
        "answer_relevancy":  ["answer_relevancy", "response_relevancy"],
        "context_precision": ["llm_context_precision_without_reference", "context_precision"],
        "context_recall":    ["context_recall", "llm_context_recall"],
    }.items():
        for c in cands:
            if c in df.columns:
                col_map[want] = c
                break
    present = [m for m in metric_names if m in col_map]

    # 4) aggregate: overall + per category ------------------------------- #
    overall = {m: nanmean(df[col_map[m]].tolist()) for m in present}
    per_cat, per_cat_n = defaultdict(dict), {}
    for cat, sub in df.groupby("category"):
        per_cat_n[cat] = len(sub)
        for m in present:
            per_cat[cat][m] = nanmean(sub[col_map[m]].tolist())

    # 5) out-of-scope abstention ----------------------------------------- #
    oos = [(r["id"], is_abstention(generated[r["id"]][0]), generated[r["id"]][0])
           for r in out_of_scope]
    oos_rate = (sum(1 for _, ok, _ in oos if ok) / len(oos)) if oos else float("nan")

    write_results_md(present, overall, per_cat, per_cat_n, oos, oos_rate,
                     has_reference, len(answerable))
    print(f"Wrote {RESULTS_MD}")


def write_results_md(metrics, overall, per_cat, per_cat_n, oos, oos_rate,
                     has_reference, n_answerable):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = ["# RAGAS Baseline — SoccerTactics_RAG\n",
         f"_Generated {ts} · pipeline: {PIPELINE} · judge: {JUDGE_MODEL} · "
         f"{n_answerable} answerable + {len(oos)} out-of-scope questions_\n"]
    if not has_reference:
        L.append("> `context_recall` skipped — no `ground_truth` field in the eval set. "
                 "Add reference answers to enable it.\n")

    L += ["## Overall (answerable)\n", "| Metric | Score |", "| --- | --- |"]
    L += [f"| {m} | {fmt(overall[m])} |" for m in metrics] + [""]

    L += ["## By category\n",
          "| Category | " + " | ".join(metrics) + " | n |",
          "| --- |" + " --- |" * (len(metrics) + 1)]
    for cat in sorted(per_cat):
        scores = " | ".join(fmt(per_cat[cat][m]) for m in metrics)
        L.append(f"| {cat} | {scores} | {per_cat_n[cat]} |")
    L.append("")

    n_ok = sum(1 for _, ok, _ in oos if ok)
    L += ["## Out-of-scope (abstention)\n",
          f"Abstention rate: **{fmt(oos_rate)}** ({n_ok}/{len(oos)})\n",
          "| id | abstained | answer (truncated) |", "| --- | --- | --- |"]
    for qid, ok, ans in oos:
        L.append(f"| {qid} | {'yes' if ok else 'NO — fabricated'} | "
                 f"{(ans or '').replace(chr(10), ' ')[:80]} |")
    L.append("")

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(L))


if __name__ == "__main__":
    main()