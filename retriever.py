"""Semantic retriever with state tracking."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import faiss
from transformers import AutoModel, AutoTokenizer

from config import load_config
from indexing import embed_texts, load_faiss


class StateTracker:
    def __init__(self) -> None:
        self.state: Dict[str, str] = {}

    def update(self, key: str, value: str) -> None:
        if value:
            self.state[key] = value

    def resolve(self, text: str) -> str:
        def _replace(match: re.Match) -> str:
            placeholder = match.group(1)
            return self.state.get(placeholder, match.group(0))

        return re.sub(r"\[(ANSWER_\d+)\]", _replace, text)

    def __repr__(self) -> str:
        return repr(self.state)


class Retriever:
    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        metadata: Optional[List[dict]] = None,
        model_name: Optional[str] = None,
    ) -> None:
        cfg = load_config()
        self.cfg = cfg
        if index is None or metadata is None:
            index, metadata = load_faiss(cfg.index_path, cfg.meta_path)
        self.index = index
        self.metadata = metadata
        self.model_name = model_name or cfg.embed_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.index.hnsw.efSearch = cfg.faiss_ef_search

    def _encode(self, query: str):
        return embed_texts([query], self.model, self.tokenizer, self.cfg.device, self.cfg.fp16)[0]

    def search(self, query: str, k: int = 20) -> List[dict]:
        query_vec = self._encode(query).astype("float32")
        scores, idxs = self.index.search(query_vec.reshape(1, -1), k)
        hits: List[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            doc = dict(self.metadata[idx])
            doc["score"] = float(score)
            hits.append(doc)
        return hits
