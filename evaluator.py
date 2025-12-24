"""Evaluation metrics: Contain-ACC and token-level F1."""
from __future__ import annotations

import re
from typing import Iterable, Tuple


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contain_acc(pred: str, gold: str) -> bool:
    p = normalize(pred)
    g = normalize(gold)
    return g in p if g else False


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(pairs: Iterable[Tuple[str, str]]) -> Tuple[float, float]:
    """pairs: iterable of (pred, gold)."""
    contain_scores = []
    f1_scores = []
    for pred, gold in pairs:
        contain_scores.append(1.0 if contain_acc(pred, gold) else 0.0)
        f1_scores.append(token_f1(pred, gold))
    contain_avg = sum(contain_scores) / len(contain_scores) if contain_scores else 0.0
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return contain_avg, f1_avg
