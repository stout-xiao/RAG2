"""RAG^2 orchestration."""
from __future__ import annotations

from typing import Tuple

from config import load_config
from filter import LogicalGateFilter
from generator import Generator
from indexing import build_faiss_index
from preprocess import process_dataset
from retriever import Retriever, StateTracker


def prepare_index(dataset_path: str) -> None:
    """Preprocess and build FAISS index from a dataset file."""
    chunks = process_dataset(dataset_path)
    build_faiss_index(chunks)


def solve_question(
    question: str,
    retriever: Retriever | None = None,
    gate: LogicalGateFilter | None = None,
    generator: Generator | None = None,
) -> Tuple[str, list, dict]:
    cfg = load_config()
    retriever = retriever or Retriever()
    gate = gate or LogicalGateFilter()
    generator = generator or Generator()

    sub_questions = generator.decompose(question)
    evidence_pool = []
    tracker = StateTracker()

    for i, sq in enumerate(sub_questions):
        resolved_sq = tracker.resolve(sq)
        candidates = retriever.search(resolved_sq)
        # 消融实验：根据配置决定是否使用 filter
        if cfg.use_filter:
            filtered_docs = gate.logical_gate(resolved_sq, candidates)
        else:
            filtered_docs = candidates  # 不过滤，直接使用检索结果
        evidence_pool.extend(filtered_docs)
        bridge_entity = generator.extract_bridge(resolved_sq, filtered_docs)
        tracker.update(f"ANSWER_{i+1}", bridge_entity)

    final_answer = generator.grounded_generate(question, evidence_pool)
    return final_answer, evidence_pool, tracker.state


if __name__ == "__main__":
    import os
    
    # Example usage (requires prebuilt index and configured LLM callable)
    cfg = load_config()
    
    # 检查索引是否已存在，避免重复构建
    index_path = "indexes/faiss.index"
    if not os.path.exists(index_path):
        print("索引不存在，正在构建 FAISS 索引...")
        prepare_index("data/hotpotqa.json")
        print("索引构建完成！\n")
    else:
        print("索引已存在，跳过构建。\n")
    
    sample_question = "Where is the ice hockey team based that Zdeno Chára currently serving as captain of?"
    answer, evidence, state = solve_question(sample_question)
    print("Answer:", answer)
    print("State:", state)
