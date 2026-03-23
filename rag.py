"""
rag.py — Retrieval-Augmented Generation engine.

Loads the pre-built FAISS index and chunks, then uses Gemini to answer
questions grounded in the retrieved podcast transcript segments.
"""
import json
import os
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from fastembed import TextEmbedding
import google.generativeai as genai

load_dotenv()

DATA_DIR    = Path("data")
CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_FILE  = DATA_DIR / "index.faiss"
META_FILE   = DATA_DIR / "meta.json"

# Singleton caches so we don't reload on every query
_embed_model: TextEmbedding | None = None
_index:       faiss.Index | None   = None
_chunks:      list[dict] | None    = None
_meta:        dict | None          = None


# ── Setup ─────────────────────────────────────────────────────────────────────

def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to your .env file:\n"
            "  GEMINI_API_KEY=YOUR_KEY_HERE"
        )
    genai.configure(api_key=api_key)


def load_index():
    """Load (or return cached) FAISS index, chunks, metadata, and embedding model."""
    global _embed_model, _index, _chunks, _meta

    if _index is not None:
        return  # already loaded

    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "No index found. Run ingestion first:\n  python ingest.py"
        )

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        _chunks = json.load(f)

    with open(META_FILE) as f:
        _meta = json.load(f)

    _index = faiss.read_index(str(INDEX_FILE))

    _embed_model = TextEmbedding(model_name=_meta.get("embed_model", "BAAI/bge-small-en-v1.5"))

    _configure_gemini()


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed the query and return the top-k most similar transcript chunks.
    Each result includes: text, start (seconds), score, timestamp (HH:MM:SS).
    """
    load_index()

    query_vec = np.array(list(_embed_model.embed([query])), dtype="float32")
    distances, indices = _index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx].copy()
        chunk["score"]     = float(dist)
        chunk["timestamp"] = _fmt_ts(chunk["start"])
        chunk["url"]       = f"https://www.youtube.com/watch?v={_meta['video_id']}&t={int(chunk['start'])}s"
        results.append(chunk)

    return results


# ── Generation ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant that answers questions strictly based on transcript excerpts from a Lex Fridman podcast.
- Only use information present in the provided transcript excerpts.
- If the answer isn't in the excerpts, say so honestly.
- Be concise but thorough.
- Reference speaker names and timestamps when relevant.
"""

def answer(query: str, k: int = 5) -> dict:
    """
    Full RAG: retrieve top-k chunks, build context, call Gemini, return answer + sources.

    Returns:
        {
            "query":   str,
            "answer":  str,
            "sources": list[dict],   # top-k retrieved chunks
        }
    """
    sources = retrieve(query, k=k)

    context_blocks = []
    for i, src in enumerate(sources, 1):
        context_blocks.append(
            f"[Excerpt {i} — {src['timestamp']}]\n{src['text']}"
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Transcript Excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    model = genai.GenerativeModel("gemini-flash-latest")
    response = model.generate_content(prompt)

    return {
        "query":   query,
        "answer":  response.text.strip(),
        "sources": sources,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_meta() -> dict:
    """Return podcast metadata (video id, url, chunk count, etc.)."""
    load_index()
    return _meta
