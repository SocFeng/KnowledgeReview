"""集中管理项目配置，所有可调参数都从环境变量 / .env 文件读取。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== DashScope =====
    dashscope_api_key: str = Field(..., alias="DASHSCOPE_API_KEY")

    llm_model: str = Field("qwen-plus", alias="LLM_MODEL")

    # Embedding：dashscope（在线 API）或 huggingface（本地模型）
    embedding_provider: str = Field("dashscope", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field("text-embedding-v3", alias="EMBEDDING_MODEL")

    # 本地 HuggingFace embedding（仅 embedding_provider=huggingface 生效）
    hf_embedding_model: str = Field(
        "BAAI/bge-large-zh-v1.5", alias="HF_EMBEDDING_MODEL"
    )
    hf_embedding_device: str = Field("auto", alias="HF_EMBEDDING_DEVICE")
    hf_embedding_batch_size: int = Field(16, alias="HF_EMBEDDING_BATCH_SIZE")
    hf_embedding_query_instruction: str = Field(
        "", alias="HF_EMBEDDING_QUERY_INSTRUCTION"
    )

    rerank_model: str = Field("gte-rerank", alias="RERANK_MODEL")

    # ===== Chunking =====
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")

    # ===== Retrieval =====
    vector_top_k: int = Field(5, alias="VECTOR_TOP_K")
    bm25_top_k: int = Field(5, alias="BM25_TOP_K")
    rerank_top_n: int = Field(4, alias="RERANK_TOP_N")

    # ===== Paths =====
    data_dir: Path = Field(PROJECT_ROOT / "data", alias="DATA_DIR")
    storage_dir: Path = Field(PROJECT_ROOT / "storage", alias="STORAGE_DIR")
    chroma_collection: str = Field("knowledge_base", alias="CHROMA_COLLECTION")

    # ===== API =====
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")

    # 派生路径
    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def docstore_dir(self) -> Path:
        """BM25 需要持久化 docstore，单独放一个目录。"""
        return self.storage_dir / "docstore"

    @property
    def chat_history_dir(self) -> Path:
        """对话记录持久化目录，每个会话一个 JSON 文件。"""
        return self.data_dir / "chat_history"


_settings: Settings | None = None


def get_settings() -> Settings:
    """单例方式返回配置；首次调用时校验环境变量。"""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
        _settings.storage_dir.mkdir(parents=True, exist_ok=True)
        _settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _settings.docstore_dir.mkdir(parents=True, exist_ok=True)
        _settings.chat_history_dir.mkdir(parents=True, exist_ok=True)
    return _settings
