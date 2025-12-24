"""Embedding and FAISS indexing."""
from __future__ import annotations

import json
from typing import List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import load_config


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts: List[str], model, tokenizer, device: str, fp16: bool = True) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    batch_size = 8
    model.to(device)
    if device.startswith("cuda") and fp16:
        model.half()
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().float().numpy())
    return np.vstack(embeddings)


def build_faiss_index(chunks: List[dict]) -> Tuple[faiss.Index, List[dict]]:
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.embed_model)
    model = AutoModel.from_pretrained(cfg.embed_model)

    vectors = embed_texts([c["text"] for c in chunks], model, tokenizer, cfg.device, cfg.fp16)
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, cfg.faiss_m)
    index.hnsw.efSearch = cfg.faiss_ef_search
    index.add(vectors.astype("float32"))

    faiss.write_index(index, cfg.index_path)
    with open(cfg.meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return index, chunks


def load_faiss(index_path: str, meta_path: str) -> Tuple[faiss.Index, List[dict]]:
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata
