"""Cloud generator wrapper for RAG^2."""
from __future__ import annotations

import json
import re
import time
from typing import Callable, List, Optional

from config import load_config
from openai_client import call_mimo_api


LLMFn = Callable[[str, Optional[str]], str]


class Generator:
    def __init__(self, call_fn: Optional[LLMFn] = None) -> None:
        cfg = load_config()
        self.cfg = cfg
        # Use call_mimo_api as default if no custom function provided
        self.call_fn = call_fn or call_mimo_api
        self.system_prompt = (
            "You are an expert planner for multi-hop information retrieval."
        )
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
            "Your task is to decompose a complex question into a sequence of simple, single-hop sub-questions that can be answered by retrieving documents.\n\n"
            "IMPORTANT: You MUST always decompose into at least 2 sub-questions, even if the question seems simple.\n"
            "The goal is planning, not answering.\n"
            "You must NOT answer the question.\n"
            "You must NOT use any external knowledge.\n\n"
            "Instructions:\n"
            "- Decompose the question into an ordered list of sub-questions (minimum 2).\n"
            "- Each sub-question should be as simple and self-contained as possible.\n"
            "- The answer to sub-question i should help answer sub-question i+1.\n"
            "- If a sub-question depends on the answer of a previous one, use the placeholder \"[ANSWER_1]\", \"[ANSWER_2]\", etc.\n"
            "- Do not include explanations, reasoning steps, or additional text.\n\n"
            "Example 1:\n"
            "Question: What is the birth city of the director of the film Inception?\n"
            "Output: {\"decomposed_questions\": [\"Who is the director of the film Inception?\", \"What is the birth city of [ANSWER_1]?\"]}\n\n"
            "Example 2:\n"
            "Question: Where is the university located that the founder of Microsoft attended?\n"
            "Output: {\"decomposed_questions\": [\"Who is the founder of Microsoft?\", \"Which university did [ANSWER_1] attend?\", \"Where is [ANSWER_2] located?\"]}\n\n"
            "Output format:\n"
            "Return ONLY a single JSON object with one key \"decomposed_questions\", whose value is a list of strings. No other text.\n\n"
            f"Question:\n{question}\n\n"
            "Output:"
        )
        try:
            raw = self._call(template, self.system_prompt).strip()
            # Remove potential markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            
            # Try to parse as JSON with "decomposed_questions" key
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "decomposed_questions" in parsed:
                sub_questions = parsed["decomposed_questions"]
            elif isinstance(parsed, list):
                sub_questions = parsed
            else:
                sub_questions = [question]
            
            if isinstance(sub_questions, list) and len(sub_questions) > 0:
                return [str(s) for s in sub_questions]
        except Exception as e:
            print(f"Warning: Failed to decompose question: {e}")
            pass
        return [question]

    def grounded_generate(self, question: str, evidence: List[dict]) -> str:
        context_blocks = []
        for i, doc in enumerate(evidence):
            context_blocks.append(f"Document {i+1}:\nTitle: {doc.get('doc_title','')}\nContent: {doc.get('text','')}")
        context = "\n\n".join(context_blocks)
        
        prompt = (
            "You are a factual question answering system.\n"
            "Answer the question strictly based on the provided documents.\n"
            "Do not use any external knowledge or assumptions.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Output ONLY the answer itself, nothing else.\n"
            "- Do NOT include phrases like 'Based on the documents', 'According to', 'The answer is', etc.\n"
            "- Do NOT provide explanations or reasoning.\n"
            "- Keep the answer as SHORT as possible (ideally 1-5 words).\n"
            "- If the answer is a name, date, number, or entity, output just that.\n"
            "- If you cannot find the answer in the documents, output only: 'unknown'\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer (output ONLY the answer, no explanation):"
        )
        if not self.call_fn:
            steps = "; ".join([doc.get("doc_title", "") for doc in evidence][:2])
            guess = evidence[0]["text"] if evidence else ""
            return f"Steps: {steps}\nFinal Answer: {guess}"
        
        system = "You are a factual question answering system."
        return self._call(prompt, system)

    def extract_bridge(self, sub_question: str, docs: List[dict]) -> str:
        best_text = docs[0]["text"] if docs else ""
        if not self.call_fn:
            match = re.search(r"([A-Z][a-zA-Z0-9_-]{2,})", best_text)
            return match.group(1) if match else best_text.split(" ")[0:3][0] if best_text else ""

        prompt = (
            "Extract the key entity from the evidence that can be used for the next retrieval hop.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Output ONLY the entity name, nothing else.\n"
            "- Do NOT include explanations, reasoning, or additional text.\n"
            "- The entity should be a person name, place name, organization, date, or other proper noun.\n"
            "- Keep it as SHORT as possible (1-5 words maximum).\n\n"
            f"Sub-question: {sub_question}\n"
            f"Evidence: {best_text}\n\n"
            "Bridge entity (output ONLY the entity):"
        )
        try:
            return self._call(prompt, self.system_prompt).strip()
        except Exception:
            return ""
