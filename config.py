"""Global configuration for RAG^2."""
from dataclasses import dataclass, field
import os
import torch


@dataclass
class AppConfig:
    api_base: str = os.getenv("GENERATOR_API_BASE", "")
    api_key: str = os.getenv("GENERATOR_API_KEY", "")
    lambda_weight: float = 0.5
    tau_min: float = 0.1
    hotpot_chunk_size: int = 4
    hotpot_chunk_overlap: int = 1
    musique_chunk_size: int = 3
    musique_chunk_overlap: int = 1
    embed_model: str = os.getenv("EMBED_MODEL", "facebook/contriever-msmarco")
    nli_model: str = os.getenv("NLI_MODEL", "microsoft/deberta-large-mnli")
    faiss_m: int = 32
    faiss_ef_search: int = 64
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    fp16: bool = True
    rate_limit_per_min: int = 30
    index_path: str = "faiss.index"
    meta_path: str = "faiss_meta.json"


def load_config() -> AppConfig:
    return AppConfig()
