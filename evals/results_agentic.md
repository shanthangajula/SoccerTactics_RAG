# RAGAS Baseline — SoccerTactics_RAG

_Generated 2026-06-17 14:00 · pipeline: agentic · judge: gpt-4o-mini · 16 answerable + 4 out-of-scope questions_

> `context_recall` skipped — no `ground_truth` field in the generations. Add references to the eval set and re-run both phases to enable it.

## Overall (answerable)

| Metric | Score |
| --- | --- |
| faithfulness | 0.838 |
| answer_relevancy | 0.532 |
| context_precision | 0.809 |

## By category

| Category | faithfulness | answer_relevancy | context_precision | n |
| --- | --- | --- | --- | --- |
| comparison | 0.845 | 0.630 | 0.750 | 4 |
| conceptual_explanation | 0.964 | 0.557 | 0.737 | 4 |
| factual_lookup | 0.792 | 0.570 | 0.750 | 4 |
| scenario_application | 0.751 | 0.371 | 1.000 | 4 |

## Out-of-scope (abstention)

Abstention rate: **1.000** (4/4)

| id | abstained | answer (truncated) |
| --- | --- | --- |
| q17 | yes | The context provided does not contain any information about the 2026 UEFA Champi |
| q18 | yes | The context only provides information about the most expensive **goalkeeper** tr |
| q19 | yes | I don't know. The provided passages do not contain information about Cristiano R |
| q20 | yes | I don't know. The provided passages are about soccer tactics and formations, and |
