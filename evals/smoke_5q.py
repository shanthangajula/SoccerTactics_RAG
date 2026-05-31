"""
End-to-end smoke test for the full agentic graph.

Runs 5 questions chosen to exercise different paths through the graph:
  1. A clean, well-supported question — should pass grader + checker on first try
  2. A vaguely-phrased question — should trigger the rewriter
  3. A specific factual question — common case, single pass
  4. A question pushing at corpus boundaries — may trigger the checker
  5. An out-of-scope question — exercises the honest "don't know" path

Reports which nodes fired for each question so you can verify nothing's broken
and the retry logic engages when it should.

Usage:
    python evals/smoke_5q.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentic_rag import ask  # noqa: E402

QUESTIONS = [
    ("clean",      "What is gegenpressing?"),
    ("vague",      "Tell me about that style where teams keep the ball with lots of short passes"),
    ("specific",   "Which manager developed tiki-taka at Barcelona, and during what years?"),
    ("boundary",   "Why did Bayern beat Barcelona 4-0 in the 2013 Champions League semi-final?"),
    ("oos",        "Who won the 2024 Champions League final?"),
]


def main() -> int:
    load_dotenv()
    rows = []

    for tag, q in QUESTIONS:
        print(f"\n{'#' * 70}")
        print(f"#  [{tag}] {q}")
        print("#" * 70)
        t0 = time.time()
        try:
            final = ask(q)
            elapsed = time.time() - t0
            rewrites = max(0, final["attempts"] - 1)
            regens = max(0, final["generate_attempts"] - 1)
            rows.append({
                "tag": tag,
                "question": q,
                "retrieve_attempts": final["attempts"],
                "rewrites": rewrites,
                "generate_attempts": final["generate_attempts"],
                "regens": regens,
                "grounded": final["grounded"],
                "answer_preview": final["answer"].strip()[:160],
                "elapsed_sec": round(elapsed, 2),
                "error": None,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"tag": tag, "question": q, "error": str(exc)})

    # ---------- summary ----------
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    header = f"{'tag':<10} {'rewrites':<9} {'regens':<7} {'grounded':<9} {'elapsed':<9}  question"
    print(header)
    print("-" * 80)
    for r in rows:
        if r.get("error"):
            print(f"{r['tag']:<10} ERROR: {r['error']}")
            continue
        print(
            f"{r['tag']:<10} {r['rewrites']:<9} {r['regens']:<7} "
            f"{str(r['grounded']):<9} {r['elapsed_sec']}s     "
            f"{r['question'][:50]}{'...' if len(r['question']) > 50 else ''}"
        )

    # Sanity flags
    print("\nSanity checks:")
    triggered_rewrite = any(r.get("rewrites", 0) > 0 for r in rows if not r.get("error"))
    triggered_regen = any(r.get("regens", 0) > 0 for r in rows if not r.get("error"))
    print(f"  At least one rewrite fired:    {triggered_rewrite}  "
          f"({'good — rewriter is reachable' if triggered_rewrite else 'note: no rewrite path exercised in this run'})")
    print(f"  At least one regeneration fired: {triggered_regen}  "
          f"({'good — checker is reachable' if triggered_regen else 'note: no regen path exercised in this run'})")
    print(f"  All queries returned non-empty: "
          f"{all(r.get('answer_preview') for r in rows if not r.get('error'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
