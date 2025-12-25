"""Global configuration for RAG^2."""
from dataclasses import dataclass, field
import os
import torch
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()


@dataclass
class AppConfig:
    api_base: str = os.getenv("GENERATOR_API_BASE", "https://api.xiaomimimo.com/v1")
    api_key: str = os.getenv("MIMO_API_KEY", "API KEY")
    model_name: str = os.getenv("MODEL_NAME", "mimo-v2-flash")
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
    index_dir: str = "indexes"
    index_path: str = "indexes/faiss.index"
    meta_path: str = "indexes/faiss_meta.json"


def load_config() -> AppConfig:
    return AppConfig()
