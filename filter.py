"""Adaptive logical gate using NLI."""
from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import load_config


class LogicalGateFilter:
    def __init__(self, model_name: str | None = None) -> None:
        cfg = load_config()
        self.cfg = cfg
        self.model_name = model_name or cfg.nli_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(cfg.device)
        if cfg.device.startswith("cuda") and cfg.fp16:
            self.model.half()
        self.model.eval()
        self.entail_idx = 2  # DeBERTa MNLI label order

    def _score(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        return float(probs[0, self.entail_idx].item())

    def logical_gate(self, hypothesis: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return []
        scores = [self._score(c["text"], hypothesis) for c in candidates]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)
        std = var**0.5
        tau = max(self.cfg.tau_min, mean + 0.5 * std)

        decorated = []
        for c, s in zip(candidates, scores):
            doc = dict(c)
            doc["entail_score"] = s
            doc["low_confidence"] = False
            if s >= tau:
                decorated.append(doc)

        if decorated:
            return sorted(decorated, key=lambda x: x["entail_score"], reverse=True)

        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        fallback = dict(candidates[best_idx])
        fallback["entail_score"] = scores[best_idx]
        fallback["low_confidence"] = True
        return [fallback]
