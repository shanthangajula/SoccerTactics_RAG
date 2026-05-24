
# Evaluation harness

Measures the soccer-tactics RAG against a small set of ground-truth questions.

## Why these metrics

A RAG can fail in two distinct ways:

1. **Retrieval failure** — the right chunks never get pulled from the vector DB
2. **Generation failure** — the right chunks arrive, but the LLM ignores them or hallucinates

Different metrics catch different failure modes, so we measure three things in tension:

| Metric | What it measures | Catches |
|---|---|---|
| `keyword_recall` | Fraction of expected keywords present in the answer | Total wrong-track answers |
| `topic_hit` | Did the retriever surface a chunk from the expected source article? | Pure retrieval failures |
| `faithfulness` (LLM-judge) | Is every claim in the answer supported by the retrieved context? | Hallucination |

If `topic_hit` is high but `faithfulness` is low → the LLM is hallucinating despite having the right context. If `topic_hit` is low → retrieval is broken; the LLM didn't have a chance.

## Files

- `ground_truth.json` — 10 questions with expected keywords, expected source topics, and reference answers
- `run_eval.py` — runs the harness, prints aggregates, optionally writes detailed JSON

## Running

```bash
# Cheap deterministic run (no extra API calls beyond the RAG itself)
python evals/run_eval.py

# With LLM-as-judge faithfulness scoring (uses gpt-4o-mini, ~$0.001 total)
python evals/run_eval.py --judge

# Save full per-question results for inspection / future regression baselines
python evals/run_eval.py --judge --out evals/results.json
```

## Interpreting output

```
SUMMARY (7/10 succeeded)
  keyword_recall  (mean): 0.525
  topic_hit       (mean): 0.600
  faithfulness    (mean): 0.970
  latency_sec     (mean): 5.23
```

These numbers are the **baseline for naive RAG.** When we add re-ranking, query
expansion, or agentic retrieval later, we re-run the harness and check whether
the metrics move in the right direction. That's the value of having evals at all
— without them, "improvements" are vibes.

## Known limitations

- **Tiny dataset.** 10 questions is enough to catch obvious regressions, not enough to detect subtle drift. For real evaluation you'd want 100+ questions covering edge cases (multi-hop, adversarial, out-of-distribution).
- **Keyword recall is brittle.** "Pep Guardiola" vs "Guardiola" vs "Josep Guardiola" all express the same fact; substring matching misses variants. Acceptable as a fast first-pass; the LLM-judge handles the nuance.
- **Single judge model.** Using one LLM to judge another correlates errors. Production setups use multiple judges or human spot-checks on judge disagreements.
- **No retrieval-only metrics.** We don't measure `context_precision` or `MRR` directly. The `topic_hit` proxy catches "did we get *anything* useful" but not "did we get the *most relevant* chunk first."