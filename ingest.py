"""
ingest.py — Download & index a Lex Fridman podcast transcript.

Usage:
    python ingest.py
    python ingest.py --video-id <YOUTUBE_VIDEO_ID>

Default video: Lex Fridman + Elon Musk (DxREm3s1scA)
"""
import os
import json
import argparse
import re
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from fastembed import TextEmbedding
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_VIDEO_ID = "DxREm3s1scA"   # Lex Fridman + Elon Musk, ~3 hrs
CHUNK_SIZE       = 500             # characters per chunk
CHUNK_OVERLAP    = 100             # overlap between consecutive chunks
DATA_DIR         = Path("data")
CHUNKS_FILE      = DATA_DIR / "chunks.json"
INDEX_FILE       = DATA_DIR / "index.faiss"
META_FILE        = DATA_DIR / "meta.json"
EMBED_MODEL      = "BAAI/bge-small-en-v1.5"


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> list[dict]:
    """Return list of {text, start, duration} dicts from YouTube."""
    console.print(f"[cyan]Fetching transcript for video:[/] [bold]{video_id}[/]")
    transcript_list = YouTubeTranscriptApi().list(video_id)
    raw_transcript = transcript_list.find_transcript(['en']).fetch()
    # Compatibility bridging for old versions returning objects
    transcript = [
        {"text": t.text, "start": t.start, "duration": t.duration} 
        if not isinstance(t, dict) else t 
        for t in raw_transcript
    ]
    console.print(f"[green]✓[/] Got [bold]{len(transcript)}[/] transcript segments")
    return transcript


def clean_text(text: str) -> str:
    """Remove filler artefacts from auto-generated captions."""
    text = re.sub(r"\[.*?\]", "", text)          # [Music], [Laughter], etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_transcript(transcript: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Merge transcript segments into overlapping text chunks.
    Each chunk carries the start timestamp of its first segment.
    """
    chunks = []
    buffer_text = ""
    buffer_start = transcript[0]["start"]
    buffer_segments = []

    for seg in transcript:
        seg_text = clean_text(seg["text"])
        if not seg_text:
            continue

        if not buffer_segments:
            buffer_start = seg["start"]

        buffer_text += " " + seg_text
        buffer_segments.append(seg)

        if len(buffer_text) >= chunk_size:
            chunks.append({
                "text":  buffer_text.strip(),
                "start": buffer_start,
            })
            # slide window back by overlap characters
            overlap_text = buffer_text[-overlap:].strip()
            buffer_text = overlap_text
            buffer_start = buffer_segments[-1]["start"]
            buffer_segments = []

    # flush remainder
    if buffer_text.strip():
        chunks.append({"text": buffer_text.strip(), "start": buffer_start})

    return chunks


def format_timestamp(seconds: float) -> str:
    """Convert seconds → HH:MM:SS string."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Generate embeddings for all chunks using fastembed (ONNX, no PyTorch)."""
    console.print(f"\n[cyan]Loading embedding model:[/] [bold]{EMBED_MODEL}[/]")
    model = TextEmbedding(model_name=EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    console.print(f"[cyan]Embedding[/] [bold]{len(texts)}[/] chunks…")

    embeddings = list(tqdm(
        model.embed(texts),
        total=len(texts),
        desc="Embedding",
        unit="chunk",
    ))
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat L2 FAISS index (exact search, great for <100k chunks)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save(chunks: list[dict], index: faiss.Index, video_id: str, embedding_dim: int):
    DATA_DIR.mkdir(exist_ok=True)

    # chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # FAISS index
    faiss.write_index(index, str(INDEX_FILE))

    # metadata
    meta = {
        "video_id":      video_id,
        "video_url":     f"https://www.youtube.com/watch?v={video_id}",
        "num_chunks":    len(chunks),
        "embedding_dim": embedding_dim,
        "embed_model":   EMBED_MODEL,
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    console.print(Panel(
        f"[green]✅ Ingestion complete![/]\n\n"
        f"• Chunks: [bold]{len(chunks)}[/]\n"
        f"• Index:  [bold]{INDEX_FILE}[/]\n"
        f"• Video:  [bold]https://www.youtube.com/watch?v={video_id}[/]",
        title="Done",
        border_style="green",
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest a YouTube podcast for RAG")
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID,
                        help="YouTube video ID (default: Lex + Elon Musk)")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]RAG Bot — Podcast Ingestion[/]\n"
        "Fetching transcript → chunking → embedding → storing FAISS index",
        border_style="cyan",
    ))

    transcript = fetch_transcript(args.video_id)
    chunks     = chunk_transcript(transcript, CHUNK_SIZE, CHUNK_OVERLAP)
    console.print(f"[green]✓[/] Created [bold]{len(chunks)}[/] chunks")

    embeddings = embed_chunks(chunks)
    index      = build_faiss_index(embeddings)

    save(chunks, index, args.video_id, embeddings.shape[1])


if __name__ == "__main__":
    main()
