"""
Agentic RAG over the soccer-tactics corpus — full CRAG loop.

Graph topology:

                       ┌── relevant ──────────────────────────────┐
    retrieve ──► grade │                                          ▼
                       └── irrelevant ──► rewrite ──► retrieve    │
                            (max 1)                               │
                                                                  │
                                       ┌──────────────────────────┘
                                       ▼
                                   generate
                                       │
                                       ▼
                                   check_hallucination
                                       │
                       ┌── grounded ───┴────── ungrounded ──┐
                       ▼                                     │
                      END                            (regen_attempts < 2)
                                                             │
                                                             ▼
                                                         generate
                                                             │
                                                             ▼
                                                   check_hallucination
                                                             │
                       (after second attempt: always END, with grounded flag)

Two independent self-checks:
  - grader (pre-generation):   "are these chunks relevant?"  -> rewrite query
  - checker (post-generation): "is the answer grounded?"     -> regenerate

Bounded retries on both paths (max 1 rewrite, max 1 regenerate) — agentic loops
without bounds are how you get $200 API bills from oscillating agents.

Usage:
    python agentic_rag.py "What is gegenpressing?"
    python agentic_rag.py "Compare 4-3-3 to 4-2-3-1" --trace
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph

REPO_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION = "soccer_tactics"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-6"

TOP_K = 5
MAX_RETRIEVE_ATTEMPTS = 2   # original + at most one rewrite
MAX_GENERATE_ATTEMPTS = 2   # original + at most one regenerate


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class GraphState(TypedDict):
    """Shared state threaded through every node.

    History fields (verdicts, query_history, answer_history) accumulate across
    attempts so the full trace is debuggable downstream.
    """
    original_question: str
    current_query: str

    # Retrieve+grade loop
    attempts: int
    documents: list[Document]
    grader_verdicts: list[dict]
    query_history: list[str]

    # Generate+check loop
    generate_attempts: int
    answer: str
    answer_history: list[str]
    hallucination_verdicts: list[dict]
    grounded: bool


# --------------------------------------------------------------------------- #
# Shared resources
# --------------------------------------------------------------------------- #

def _retriever() -> Chroma:
    if not CHROMA_DIR.exists():
        sys.exit(f"No index at {CHROMA_DIR}. Run `python ingestion/index.py` first.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def _llm(max_tokens: int = 512) -> ChatAnthropic:
    return ChatAnthropic(model=CHAT_MODEL, temperature=0, max_tokens=max_tokens)


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        topic = d.metadata.get("topic", "unknown")
        blocks.append(f"[{i}] topic={topic}\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def _parse_json_verdict(raw: str, key: str, default_true: bool = True) -> dict:
    """Parse the LLM's JSON verdict defensively.

    Returns {key: bool, "reason": str, "raw": str}. Defaults to `default_true`
    on parse failure — better to over-emit than to silently block everything
    when the model has a bad-parse day.
    """
    verdict = {key: default_true, "reason": "parse_failure_default", "raw": raw}
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
        verdict_value = str(parsed.get(key, "")).strip().lower()
        verdict[key] = verdict_value.startswith("y")
        verdict["reason"] = str(parsed.get("reason", "")).strip()
        verdict["raw"] = raw
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    return verdict


# --------------------------------------------------------------------------- #
# Node: retrieve
# --------------------------------------------------------------------------- #

def node_retrieve(state: GraphState) -> dict:
    db = _retriever()
    docs = db.similarity_search(state["current_query"], k=TOP_K)
    return {
        "documents": docs,
        "attempts": state["attempts"] + 1,
    }


# --------------------------------------------------------------------------- #
# Node: grade (relevance check on retrieved chunks)
# --------------------------------------------------------------------------- #

GRADER_SYSTEM = (
    "You are a strict relevance grader for a retrieval-augmented generation system. "
    "Given a USER QUESTION and a set of retrieved CONTEXT passages, decide whether "
    "the passages collectively contain enough information to answer the question. "
    "Be honest — passages that are only tangentially related should be marked "
    "irrelevant. A single highly-relevant passage is enough to mark relevant. "
    "Respond with ONLY a JSON object: "
    '{"relevant": "yes" | "no", "reason": "<one short sentence>"}'
)

GRADER_USER_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

Is this context sufficient to answer the question?"""


def node_grade(state: GraphState) -> dict:
    context = _format_context(state["documents"])
    msg = _llm(max_tokens=200).invoke([
        ("system", GRADER_SYSTEM),
        ("user", GRADER_USER_TEMPLATE.format(
            question=state["original_question"],
            context=context,
        )),
    ])
    raw = msg.content if isinstance(msg.content, str) else str(msg.content)

    parsed = _parse_json_verdict(raw, key="relevant", default_true=True)
    verdict = {
        "relevant": "yes" if parsed["relevant"] else "no",
        "reason": parsed["reason"],
        "raw": parsed["raw"],
    }
    print(f"  [grade]   relevant={verdict['relevant']}  reason={verdict['reason']!r}")
    return {"grader_verdicts": state["grader_verdicts"] + [verdict]}


# --------------------------------------------------------------------------- #
# Node: rewrite
# --------------------------------------------------------------------------- #

REWRITER_SYSTEM = (
    "You rewrite user questions to improve retrieval from a vector database of "
    "soccer-tactics articles. You will be told why the previous retrieval failed. "
    "Produce a SINGLE improved query — more specific, using terminology likely to "
    "appear in encyclopedia articles, removing ambiguity. Do not answer the "
    "question. Reply with ONLY the rewritten query, no preamble, no quotes."
)

REWRITER_USER_TEMPLATE = """ORIGINAL QUESTION: {question}

PREVIOUS QUERY THAT FAILED: {previous_query}

WHY IT FAILED (grader's reason): {reason}

Rewrite the query."""


def node_rewrite(state: GraphState) -> dict:
    last_verdict = state["grader_verdicts"][-1] if state["grader_verdicts"] else {}
    reason = last_verdict.get("reason", "no specific reason given")

    msg = _llm(max_tokens=150).invoke([
        ("system", REWRITER_SYSTEM),
        ("user", REWRITER_USER_TEMPLATE.format(
            question=state["original_question"],
            previous_query=state["current_query"],
            reason=reason,
        )),
    ])
    new_query = (msg.content if isinstance(msg.content, str) else str(msg.content)).strip().strip('"')
    print(f"  [rewrite] {state['current_query']!r}  -->  {new_query!r}")
    return {
        "current_query": new_query,
        "query_history": state["query_history"] + [new_query],
    }


# --------------------------------------------------------------------------- #
# Node: generate
# --------------------------------------------------------------------------- #

GENERATOR_SYSTEM = (
    "You are a soccer tactics analyst. Answer the user's question using ONLY the "
    "passages in <context>. If the passages don't contain the answer, say you "
    "don't know — do not invent details. Be concrete and concise. When useful, "
    "cite the topic in brackets like [tiki-taka]."
)

GENERATOR_USER_TEMPLATE = """<context>
{context}
</context>

Question: {question}

Answer:"""

REGENERATOR_USER_TEMPLATE = """<context>
{context}
</context>

Question: {question}

Your previous attempt was rejected by a faithfulness check for the following reason:
  {hallucination_reason}

Previous (rejected) answer:
  {previous_answer}

Write a new answer that stays strictly within the context. If the context does
not support a confident answer, say so explicitly rather than inventing details.

Answer:"""


def node_generate(state: GraphState) -> dict:
    """Generate an answer. On the regeneration pass, surface the previous
    attempt and the checker's complaint so the model can self-correct."""
    context = _format_context(state["documents"])
    generate_attempts = state["generate_attempts"] + 1

    is_regen = generate_attempts > 1 and state["hallucination_verdicts"]
    if is_regen:
        last_verdict = state["hallucination_verdicts"][-1]
        user_msg = REGENERATOR_USER_TEMPLATE.format(
            context=context,
            question=state["original_question"],
            hallucination_reason=last_verdict.get("reason", "ungrounded claims"),
            previous_answer=state["answer"],
        )
        print(f"  [regen]   regenerating (attempt {generate_attempts})")
    else:
        user_msg = GENERATOR_USER_TEMPLATE.format(
            context=context,
            question=state["original_question"],
        )

    msg = _llm(max_tokens=1024).invoke([
        ("system", GENERATOR_SYSTEM),
        ("user", user_msg),
    ])
    answer = msg.content if isinstance(msg.content, str) else str(msg.content)
    return {
        "answer": answer,
        "answer_history": state["answer_history"] + [answer],
        "generate_attempts": generate_attempts,
    }


# --------------------------------------------------------------------------- #
# Node: hallucination check (post-generation groundedness)
# --------------------------------------------------------------------------- #

CHECKER_SYSTEM = (
    "You are a strict faithfulness checker. Given a set of CONTEXT passages and "
    "an ANSWER produced by an AI system, decide whether every factual claim in "
    "the answer is supported by the context. An answer that says 'I don't know' "
    "or appropriately acknowledges missing information is GROUNDED. An answer "
    "that adds plausible-sounding specifics not in the context (dates, scores, "
    "names, causal claims) is UNGROUNDED. Be strict — partial support is not "
    "support. Respond with ONLY a JSON object: "
    '{"grounded": "yes" | "no", "reason": "<one short sentence>"}'
)

CHECKER_USER_TEMPLATE = """CONTEXT:
{context}

ANSWER:
{answer}

Is every factual claim in the answer supported by the context?"""


def node_check_hallucination(state: GraphState) -> dict:
    context = _format_context(state["documents"])
    msg = _llm(max_tokens=200).invoke([
        ("system", CHECKER_SYSTEM),
        ("user", CHECKER_USER_TEMPLATE.format(
            context=context,
            answer=state["answer"],
        )),
    ])
    raw = msg.content if isinstance(msg.content, str) else str(msg.content)

    parsed = _parse_json_verdict(raw, key="grounded", default_true=True)
    verdict = {
        "grounded": "yes" if parsed["grounded"] else "no",
        "reason": parsed["reason"],
        "raw": parsed["raw"],
    }
    print(f"  [check]   grounded={verdict['grounded']}  reason={verdict['reason']!r}")
    return {
        "hallucination_verdicts": state["hallucination_verdicts"] + [verdict],
        "grounded": parsed["grounded"],
    }


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #

def route_after_grade(state: GraphState) -> str:
    """Relevant chunks -> generate. Irrelevant + retries left -> rewrite. Else -> generate anyway."""
    last = state["grader_verdicts"][-1]
    if last.get("relevant") == "yes":
        return "generate"
    if state["attempts"] >= MAX_RETRIEVE_ATTEMPTS:
        print(f"  [route]   out of retrieve retries (attempts={state['attempts']}), generating anyway")
        return "generate"
    print(f"  [route]   irrelevant, attempt {state['attempts']}/{MAX_RETRIEVE_ATTEMPTS}, rewriting")
    return "rewrite"


def route_after_check(state: GraphState) -> str:
    """Grounded -> end. Ungrounded + retries left -> regenerate. Else -> end with flag."""
    if state["grounded"]:
        return "end"
    if state["generate_attempts"] >= MAX_GENERATE_ATTEMPTS:
        print(f"  [route]   out of regen retries (generate_attempts={state['generate_attempts']}), returning with grounded=False flag")
        return "end"
    print(f"  [route]   ungrounded, attempt {state['generate_attempts']}/{MAX_GENERATE_ATTEMPTS}, regenerating")
    return "regenerate"


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", node_retrieve)
    g.add_node("grade", node_grade)
    g.add_node("rewrite", node_rewrite)
    g.add_node("generate", node_generate)
    g.add_node("check", node_check_hallucination)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", route_after_check, {
        "end": END,
        "regenerate": "generate",
    })

    return g.compile()


# --------------------------------------------------------------------------- #
# Public API + CLI
# --------------------------------------------------------------------------- #

def ask(question: str) -> dict:
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")

    graph = build_graph()
    initial: GraphState = {
        "original_question": question,
        "current_query": question,
        "attempts": 0,
        "documents": [],
        "grader_verdicts": [],
        "query_history": [question],
        "generate_attempts": 0,
        "answer": "",
        "answer_history": [],
        "hallucination_verdicts": [],
        "grounded": True,
    }
    return graph.invoke(initial)


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--trace", action="store_true",
                   help="Print the full state trace after running")
    args = p.parse_args()

    final = ask(args.question)

    print("\n" + "=" * 70)
    print(f"Q: {args.question}")
    print("=" * 70)
    print(final["answer"].strip())
    print("=" * 70)
    sources = sorted({d.metadata.get("topic", "?") for d in final["documents"]})
    print(f"Sources:           {', '.join(sources)}")
    print(f"Retrieve attempts: {final['attempts']}")
    print(f"Generate attempts: {final['generate_attempts']}")
    print(f"Grounded:          {final['grounded']}")
    print(f"Queries used:      {final['query_history']}")

    if args.trace:
        print("\n--- Grader trace ---")
        for i, v in enumerate(final["grader_verdicts"], 1):
            print(f"  attempt {i}: relevant={v.get('relevant')}  reason={v.get('reason')}")
        print("\n--- Hallucination check trace ---")
        for i, v in enumerate(final["hallucination_verdicts"], 1):
            print(f"  attempt {i}: grounded={v.get('grounded')}  reason={v.get('reason')}")
        if len(final["answer_history"]) > 1:
            print("\n--- Answer history ---")
            for i, a in enumerate(final["answer_history"], 1):
                print(f"  attempt {i}: {a[:140]}{'...' if len(a) > 140 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())