# Deployment guide

Deploy the soccer-tactics RAG as a public Streamlit app on Hugging Face Spaces. **Free tier, no credit card required.**

End result: a URL like `https://huggingface.co/spaces/shanthangajula/soccer-tactics-rag` you can put on your resume and share.

## Why HF Spaces (and not Render/Railway)

| | HF Spaces | Render | Railway |
|---|---|---|---|
| Free tier | Yes, no time cap (sleeps after 48h idle) | Yes, but 750h/mo + 90-day sleep | $5/mo credit, then paid |
| Cold start | ~30s | ~15s | ~10s |
| ML-recognizable domain | ✅ `huggingface.co` | ⚠ `onrender.com` | ⚠ `railway.app` |
| Native Streamlit support | ✅ SDK preset | Manual config | Manual config |

HF Spaces is the canonical ML demo host. Recruiters and hiring managers recognize the domain instantly.

## Prerequisites

- A Hugging Face account (sign up at https://huggingface.co/join)
- Your `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` ready to paste

## Step-by-step

### 1. Create the Space

1. Go to https://huggingface.co/new-space
2. Owner: your username
3. Space name: `soccer-tactics-rag`
4. License: MIT (or whatever your repo uses)
5. **Select SDK: Streamlit**
6. Hardware: CPU basic (free tier — fine for this)
7. Visibility: Public
8. Click "Create Space"

HF creates an empty git repo at `https://huggingface.co/spaces/<your-username>/soccer-tactics-rag`.

### 2. Add API keys as Repository Secrets

**Critical: do not commit your `.env` file.** The Space reads secrets from HF's secret store, not from a file.

1. Open your new Space → Settings tab → "Variables and secrets" section
2. Add two **secrets** (not variables):
   - Name: `ANTHROPIC_API_KEY`, Value: your real key
   - Name: `OPENAI_API_KEY`, Value: your real key
3. Save

These get injected as environment variables at container startup. Your code reads them via `os.getenv()` — the existing `load_dotenv()` call is a no-op when there's no `.env` file, which is exactly what we want on the Space.

### 3. Push your code to the Space

The Space is just another git remote. From your local `SoccerTactics_RAG` repo:

```bash
# Add the HF Space as a second remote
git remote add space https://huggingface.co/spaces/<your-username>/soccer-tactics-rag

# First push will need your HF access token as password.
# Generate one at https://huggingface.co/settings/tokens (give it "write" scope).
# When git asks for username, use your HF username; for password, paste the token.
git push space main
```

If you're on a different branch name (e.g. `master` or `InitialTrains`), push that branch to `main` on the Space:

```bash
git push space <your-branch>:main
```

### 4. Watch the build

Go to your Space's page. You'll see a "Building" status. HF reads `requirements.txt`, installs dependencies, then runs `streamlit run app.py` automatically.

First build takes 3-5 minutes (downloading and caching all the langchain wheels). Subsequent builds are faster.

If the build fails, click "Logs" to see why. Most common causes:

- Missing dependency in `requirements.txt`
- A module import that worked locally but breaks on Linux (rare with our stack)
- Missing API key secrets (you'll see a clear error in the app once it loads)

### 5. First-run index build

The first time the container boots, it will:

1. Find that `chroma_db/` doesn't exist (it's in `.gitignore`)
2. Find that `data/raw/` is empty
3. Run the scraper to fetch the 20 Wikipedia articles (~2 minutes)
4. Build the Chroma index from them (~1 minute, embeddings cost ~$0.01)
5. Cache it for the lifetime of the container

You'll see a spinner in the UI saying "Preparing the index (one-time startup cost)…" — this is normal. Subsequent queries on the same container are instant on the index side.

When the container sleeps (after 48h idle) and gets woken up, it rebuilds the index from scratch — about 3 minutes to first query.

### 6. Test it

Open your Space URL. Ask a question. You should see:

- The answer
- Retrieve / generate attempt counts
- Grounded badge
- Expandable retrieved sources and verdict traces

If the agentic mode triggers a rewrite or regeneration, you'll see a callout box explaining what happened. That's the demo gold — recruiters can literally watch the system self-correct.

## Faster cold starts (optional)

The default config rebuilds the index on every cold start. To avoid that, commit the prebuilt `chroma_db/` to git:

```bash
# Remove chroma_db from .gitignore (only the line for chroma_db/)
# Then commit the built index
git add chroma_db/
git commit -m "Commit prebuilt Chroma index for faster Space cold starts"
git push space main
```

Tradeoffs:

- ✅ Cold-start to first query drops from ~3 min to ~30 sec
- ❌ Adds ~10-50 MB of binary files to git history
- ❌ Index gets stale if you change chunking, embedding model, or corpus — you have to rebuild and re-commit

For a demo project: probably worth it. For an actively-developed system: not worth it.

## Cost expectations

Per query on HF Spaces:
- **Naive RAG**: 1 Anthropic call (~500-1000 input tokens, ~200 output) + 1 OpenAI embedding (~10 tokens) ≈ **$0.005**
- **Agentic CRAG**: 3-5 Anthropic calls (grader + generator + checker + optional rewriter + optional regenerator) ≈ **$0.02**

A hundred curious recruiters trying it out costs you ~$2. The HF Space itself is free.

To cap exposure, set a usage cap in both provider dashboards:
- Anthropic: Console → Settings → Billing → "Spend limits"
- OpenAI: Platform → Settings → Limits → "Usage limits"

A $20/month cap is plenty for demo traffic and protects you against a misbehaving agent or someone scripting the UI.

## What goes on your resume

Once deployed, the bullet becomes:

> Built and **deployed** a compound RAG system over a soccer-tactics corpus using Claude Sonnet, OpenAI embeddings, and Chroma. Implemented the full Corrective RAG (CRAG) pattern in LangGraph with document grader, query rewriter, and post-generation hallucination checker. **Live demo: huggingface.co/spaces/<your-username>/soccer-tactics-rag**

The link is the bullet. Resume readers click links — every recruiter I've worked with has confirmed this. A working demo cuts through every "can they actually build things" filter.
