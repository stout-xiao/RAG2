"""Dataset loading and scientific chunking."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from config import load_config


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter to avoid heavy deps."""
    pieces = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def make_chunks(sentences: List[str], size: int, overlap: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    step = max(1, size - overlap)
    for start in range(0, len(sentences), step):
        chunk = sentences[start : start + size]
        if chunk:
            chunks.append(chunk)
    return chunks


def infer_dataset_name(file_path: str) -> str:
    name = Path(file_path).name.lower()
    if "hotpot" in name:
        return "hotpotqa"
    if "musique" in name:
        return "musique"
    return "generic"


def _yield_documents(example: Dict, dataset_name: str) -> Iterable[Tuple[str, List[str]]]:
    """Yield (title, sentences) pairs from a raw example."""
    if "context" in example and isinstance(example["context"], list):
        for ctx in example["context"]:
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                title, paragraphs = ctx[0], ctx[1]
                if isinstance(paragraphs, list):
                    sentences = [s.strip() for s in paragraphs if s.strip()]
                else:
                    sentences = split_sentences(str(paragraphs))
                yield title, sentences
            elif isinstance(ctx, dict):
                title = ctx.get("title", "")
                raw = ctx.get("context") or ctx.get("text") or ""
                sentences = ctx.get("sentences")
                if sentences and isinstance(sentences, list):
                    sentences = [s.strip() for s in sentences if s.strip()]
                else:
                    sentences = split_sentences(str(raw))
                yield title, sentences
    if "paragraphs" in example and isinstance(example["paragraphs"], list):
        for para in example["paragraphs"]:
            title = para.get("title", "")
            raw = para.get("context") or para.get("paragraph_text") or para.get("text") or ""
            sentences = para.get("sentences")
            if sentences and isinstance(sentences, list):
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                sentences = split_sentences(str(raw))
            yield title, sentences
    if dataset_name == "generic" and "title" in example and "context" in example:
        title = str(example.get("title", ""))
        sentences = example.get("sentences")
        raw = example.get("context", "")
        if sentences and isinstance(sentences, list):
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = split_sentences(str(raw))
        yield title, sentences


def process_dataset(file_path: str) -> List[Dict]:
    cfg = load_config()
    dataset_name = infer_dataset_name(file_path)
    size = cfg.hotpot_chunk_size if dataset_name == "hotpotqa" else cfg.musique_chunk_size
    overlap = cfg.hotpot_chunk_overlap if dataset_name == "hotpotqa" else cfg.musique_chunk_overlap

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks: List[Dict] = []
    chunk_id = 0
    for example in data:
        for doc_title, sentences in _yield_documents(example, dataset_name):
            for window in make_chunks(sentences, size=size, overlap=overlap):
                chunks.append(
                    {
                        "doc_title": doc_title,
                        "chunk_id": f"{doc_title}_{chunk_id}",
                        "text": " ".join(window),
                        "source_example": example.get("_id") or example.get("id") or "",
                    }
                )
                chunk_id += 1
    return chunks
