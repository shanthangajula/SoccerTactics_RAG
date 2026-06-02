"""
Streamlit UI for the soccer-tactics RAG.

Exposes both the naive and the agentic (CRAG) modes. Shows retrieval sources,
grader and hallucination-checker verdicts, and the query history when the
agent rewrote or regenerated.

Run locally:
    streamlit run app.py

Deployed:
    HF Spaces (Streamlit SDK) — see README for deployment instructions.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make sibling modules importable
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env locally; on HF Spaces these come from Repository Secrets
load_dotenv()

st.set_page_config(
    page_title="Soccer Tactics RAG",
    page_icon="⚽",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Index bootstrap
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading vector index…")
def ensure_index() -> str:
    """Ensure the Chroma index exists. Build it on first run if missing.

    On HF Spaces this runs once per container lifecycle and is cached.
    """
    from ingestion import index as index_mod

    chroma_dir = REPO_ROOT / "chroma_db"
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        return "loaded"

    raw_dir = REPO_ROOT / "data" / "raw"
    if not raw_dir.exists() or not any(raw_dir.glob("*.txt")):
        # No raw corpus either — need to scrape first.
        from ingestion import scraper
        scraper.main()

    index_mod.build_index()
    return "built"


# --------------------------------------------------------------------------- #
# Env check
# --------------------------------------------------------------------------- #

def check_keys() -> bool:
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        st.error(
            f"Missing API keys: {', '.join(missing)}. "
            "Set them as Repository Secrets in HF Spaces settings, "
            "or in a local `.env` file."
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.title("⚽ Soccer Tactics RAG")
st.caption(
    "A compound AI workflow combining retrieval, agentic self-correction (CRAG), "
    "and evaluation over 20 Wikipedia articles on soccer tactics. "
    "[Source](https://github.com/shanthangajula/SoccerTactics_RAG)"
)

# Sidebar: mode + config
with st.sidebar:
    st.header("Configuration")
    mode = st.radio(
        "Pipeline",
        ["Agentic (CRAG)", "Naive RAG"],
        help=(
            "**Naive**: retrieve → generate.\n\n"
            "**Agentic**: retrieve → grade → (rewrite if irrelevant) → "
            "generate → check (regenerate if ungrounded). Slower but self-correcting."
        ),
    )
    st.markdown("---")
    st.markdown(
        "**Corpus:** 20 Wikipedia articles  \n"
        "**Embeddings:** OpenAI `text-embedding-3-small`  \n"
        "**Generator:** Claude Sonnet 4.6  \n"
        "**Vector DB:** Chroma (local)"
    )
    st.markdown("---")
    with st.expander("Example questions"):
        st.markdown(
            "- What is gegenpressing?\n"
            "- How does a false nine create space?\n"
            "- Compare 4-3-3 to 4-2-3-1\n"
            "- What is catenaccio?\n"
            "- Who developed tiki-taka at Barcelona?"
        )

if not check_keys():
    st.stop()

with st.spinner("Preparing the index (one-time startup cost)…"):
    status = ensure_index()
if status == "built":
    st.info("Built the index from scratch on this container. Subsequent queries will be fast.")

question = st.text_input(
    "Ask a question about soccer tactics:",
    placeholder="What is gegenpressing?",
    key="question_input",
)

go = st.button("Ask", type="primary", disabled=not question.strip())


# --------------------------------------------------------------------------- #
# Answer
# --------------------------------------------------------------------------- #

if go and question.strip():
    t0 = time.time()
    try:
        if mode == "Naive RAG":
            from rag import ask as ask_naive
            answer, sources = ask_naive(question)
            elapsed = time.time() - t0
            st.markdown("### Answer")
            st.markdown(answer)
            st.caption(f"⏱ {elapsed:.1f}s · naive pipeline")
            with st.expander(f"Retrieved sources ({len(sources)} chunks)"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}] topic: `{doc.metadata.get('topic', '?')}`**"
                    )
                    st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))
                    st.markdown("---")

        else:  # Agentic CRAG
            from agentic_rag import ask as ask_agentic
            final = ask_agentic(question)
            elapsed = time.time() - t0

            # Top-line answer
            st.markdown("### Answer")
            st.markdown(final["answer"])

            # Grounded badge
            cols = st.columns(4)
            cols[0].metric("Retrieve attempts", final["attempts"])
            cols[1].metric("Generate attempts", final["generate_attempts"])
            cols[2].metric("Grounded", "✅ Yes" if final["grounded"] else "⚠️ No")
            cols[3].metric("Elapsed", f"{elapsed:.1f}s")

            # Agentic trace
            rewrote = final["attempts"] > 1
            regenerated = final["generate_attempts"] > 1
            if rewrote or regenerated:
                with st.expander("🔍 The agent self-corrected", expanded=True):
                    if rewrote:
                        st.markdown("**Query rewrite triggered:**")
                        for i, q in enumerate(final["query_history"], 1):
                            label = "original" if i == 1 else f"rewrite {i - 1}"
                            st.markdown(f"- *{label}*: `{q}`")
                    if regenerated:
                        st.markdown("**Regeneration triggered:**")
                        st.markdown(
                            "The hallucination checker rejected the first answer; "
                            "the system regenerated with the checker's reason as corrective context."
                        )

            with st.expander(f"Grader verdicts ({len(final['grader_verdicts'])})"):
                for i, v in enumerate(final["grader_verdicts"], 1):
                    flag = "✅" if v.get("relevant") == "yes" else "❌"
                    st.markdown(f"{flag} **Attempt {i}**: relevant = `{v.get('relevant')}`")
                    st.caption(v.get("reason", ""))

            with st.expander(f"Hallucination check verdicts ({len(final['hallucination_verdicts'])})"):
                for i, v in enumerate(final["hallucination_verdicts"], 1):
                    flag = "✅" if v.get("grounded") == "yes" else "❌"
                    st.markdown(f"{flag} **Attempt {i}**: grounded = `{v.get('grounded')}`")
                    st.caption(v.get("reason", ""))

            sources = final["documents"]
            with st.expander(f"Retrieved sources ({len(sources)} chunks)"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}] topic: `{doc.metadata.get('topic', '?')}`**"
                    )
                    st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))
                    st.markdown("---")

    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
        st.exception(exc)
