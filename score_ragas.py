#!/usr/bin/env python3
"""
Phase 2 of 2 — score cached generations with RAGAS.

Runs in a SEPARATE, eval-only venv (see requirements-eval.txt): ragas 0.2.x +
langchain 0.3.x. Imports ragas; does NOT import your pipeline — that's how it
sidesteps the langchain-core 1.x (pipeline) vs 0.3.x (ragas) conflict.

Metrics:
  - faithfulness            (no reference needed)
  - answer_relevancy        (no reference needed)
  - context_precision       (LLMContextPrecisionWithoutReference; no reference)
  - context_recall          (LLMContextRecall; only if `ground_truth` present)

Out-of-scope questions are scored on abstention, not pushed through RAGAS.

Usage:
    python score_ragas.py
"""
from __future__ import annotations

import json
import math
import datetime as dt
from pathlib import Path
from collections import defaultdict

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
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --------------------------------------------------------------------------- #
PIPELINE    = "agentic"                  # must match the gen_answers.py run
GENERATIONS = Path(f"evals/generations_{PIPELINE}.jsonl")
RESULTS_MD  = Path(f"evals/results_{PIPELINE}.md")
JUDGE_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

ABSTENTION_MARKERS = [
    "don't have", "do not have", "don't know", "do not know", "i don't know",
    "not in", "knowledge base", "no information", "no relevant",
    "not contain", "does not contain", "doesn't contain",
    "not mention", "does not mention", "doesn't mention",
    "cannot answer", "can't answer", "unable to", "outside the scope",
    "not covered", "not enough information", "insufficient",
    "cannot determine", "can't determine",
]


def nanmean(xs: list[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def is_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in ABSTENTION_MARKERS)


def fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> None:
    rows = [json.loads(l) for l in GENERATIONS.read_text().splitlines() if l.strip()]
    answerable   = [r for r in rows if r.get("answerable", True)]
    out_of_scope = [r for r in rows if not r.get("answerable", True)]

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

    samples, ids, cats = [], [], []
    for r in answerable:
        kw = dict(user_input=r["question"], response=r["answer"],
                  retrieved_contexts=r["contexts"])
        if "ground_truth" in r:
            kw["reference"] = r["ground_truth"]
        samples.append(SingleTurnSample(**kw))
        ids.append(r["id"]); cats.append(r["category"])

    print(f"Scoring {len(samples)} answerable questions: {', '.join(metric_names)}")
    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics)
    df = result.to_pandas()
    df.insert(0, "id", ids)
    df.insert(1, "category", cats)

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

    overall = {m: nanmean(df[col_map[m]].tolist()) for m in present}
    per_cat, per_cat_n = defaultdict(dict), {}
    for cat, sub in df.groupby("category"):
        per_cat_n[cat] = len(sub)
        for m in present:
            per_cat[cat][m] = nanmean(sub[col_map[m]].tolist())

    oos = [(r["id"], is_abstention(r["answer"]), r["answer"]) for r in out_of_scope]
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
        L.append("> `context_recall` skipped — no `ground_truth` field in the generations. "
                 "Add references to the eval set and re-run both phases to enable it.\n")

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
