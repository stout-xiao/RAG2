"""Cloud generator wrapper for RAG^2."""
from __future__ import annotations

import json
import re
import time
from typing import Callable, List, Optional

from config import load_config


LLMFn = Callable[[str, Optional[str]], str]


class Generator:
    def __init__(self, call_fn: Optional[LLMFn] = None) -> None:
        cfg = load_config()
        self.cfg = cfg
        self.call_fn = call_fn
        self.system_prompt = "You are MiMo-V2-Flash tasked with multi-hop QA planning and grounded generation."
        self._call_times: List[float] = []

    def _throttle(self) -> None:
        if self.cfg.rate_limit_per_min <= 0:
            return
        now = time.time()
        window_start = now - 60
        self._call_times = [t for t in self._call_times if t >= window_start]
        if len(self._call_times) >= self.cfg.rate_limit_per_min:
            sleep = 60 - (now - self._call_times[0])
            if sleep > 0:
                time.sleep(sleep)
        self._call_times.append(time.time())

    def _call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.call_fn:
            raise RuntimeError("No LLM callable configured for Generator.")
        self._throttle()
        return self.call_fn(prompt, system_prompt or self.system_prompt)

    def decompose(self, question: str) -> List[str]:
        template = (
            "分解多跳问题，输出 JSON 数组，每个元素是子问题字符串。"
            "必须使用 [ANSWER_1], [ANSWER_2] 等占位符指代上一步答案。\n"
            f"问题: {question}\n"
            '示例输出: ["谁创建了[ANSWER_1]所在的公司?", "他出生在哪里?"]'
        )
        try:
            raw = self._call(template, self.system_prompt)
            sub_questions = json.loads(raw)
            if isinstance(sub_questions, list) and sub_questions:
                return [str(s) for s in sub_questions]
        except Exception:
            pass
        return [question]

    def grounded_generate(self, question: str, evidence: List[dict]) -> str:
        context_blocks = []
        for i, doc in enumerate(evidence):
            context_blocks.append(f"[{i}] {doc.get('doc_title','')} :: {doc.get('text','')}")
        context = "\n".join(context_blocks)
        prompt = (
            "仅基于 Provided Context 回答，不得凭空臆测。"
            "请列出简短推理步骤，再给出最终答案。\n"
            f"Question: {question}\n"
            f"Provided Context:\n{context}\n"
            "Answer format:\nSteps: <step1>; <step2>; ...\nFinal Answer: <text>"
        )
        if not self.call_fn:
            steps = "; ".join([doc.get("doc_title", "") for doc in evidence][:2])
            guess = evidence[0]["text"] if evidence else ""
            return f"Steps: {steps}\nFinal Answer: {guess}"
        return self._call(prompt, self.system_prompt)

    def extract_bridge(self, sub_question: str, docs: List[dict]) -> str:
        best_text = docs[0]["text"] if docs else ""
        if not self.call_fn:
            match = re.search(r"([A-Z][a-zA-Z0-9_-]{2,})", best_text)
            return match.group(1) if match else best_text.split(" ")[0:3][0] if best_text else ""

        prompt = (
            "从检索证据中提取可用于下一跳检索的锚点实体，保持简短。\n"
            f"子问题: {sub_question}\n"
            f"证据: {best_text}"
        )
        try:
            return self._call(prompt, self.system_prompt).strip()
        except Exception:
            return ""
