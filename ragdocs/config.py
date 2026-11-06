"""配置加载：把 settings.yaml 读成不可变的 Settings 对象。

设计理由：RAG 系统里有大量可调参数（分块大小、top_k、模型名……），
散落在代码里会变成「改一个值要翻五个文件」。集中到 YAML 后，
调参不需要改代码，也便于把「决策」和「实现」分离——这正是本项目的
学习重点之一。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class OllamaCfg:
    base_url: str
    embed_model: str
    llm_model: str
    request_timeout: float
    restart_cmd: str


@dataclass(frozen=True)
class EmbeddingCfg:
    backend: str          # "ollama" | "local" | "llamacpp"
    local_model: str     # local 后端用的 sentence-transformers 模型名


@dataclass(frozen=True)
class RemoteEmbedCfg:
    """远程 embedding 服务（OpenAI 兼容 /v1/embeddings 协议）。

    用于本机无可用 GPU、CPU 推理过慢的场景：把 embedding 外包给云端
    MaaS（如讯飞 MaaS 的 Qwen3-Embedding-8B）。注意：文本会出网，
    且远程模型维度（如 768）与本地 bge-m3（1024）不同，切换后需删
    data/ 全量重建索引。api_key 属敏感凭证，勿提交到版本库。
    """
    base_url: str        # 含 /embeddings 的完整地址
    api_key: str
    model: str
    timeout: float = 60.0
    dims: int = 0        # 0 = 不声明，由服务端默认（如 768）
    batch_size: int = 8  # 单次请求的文本条数；云端服务宜小，避免单请求过大超时


@dataclass(frozen=True)
class LlamaCppEmbedCfg:
    """llama.cpp（GGUF）embedding 后端配置。

    与 Ollama/sentence-transformers 不同，llama.cpp 直接读本地 GGUF 文件、
    进程内推理，无服务依赖。模型需自行准备（Ollama 的模型不是裸 GGUF，
    不能复用；需从 HuggingFace 下载如 CompendiumLabs/bge-m3-gguf）。
    """
    model_path: str      # GGUF 文件路径（支持 ~/ 展开）
    n_ctx: int = 8192    # 上下文窗口；bge-m3 支持 8k
    n_threads: int = 0   # 0 = 自动（llama.cpp 按 CPU 核数）
    n_gpu_layers: int = 0  # 0 = 纯 CPU；有 GPU 可设 -1 全卸载
    batch_size: int = 32


@dataclass(frozen=True)
class ChunkCfg:
    chunk_size: int
    chunk_overlap: int
    short_doc_threshold: int
    inject_fields: tuple[str, ...]


@dataclass(frozen=True)
class IngestCfg:
    include_ext: frozenset[str]
    exclude_dirs: frozenset[str]
    max_file_bytes: int


@dataclass(frozen=True)
class QueryCfg:
    top_k: int
    similarity_threshold: float
    system_prompt: str


@dataclass(frozen=True)
class PrivacyCfg:
    exclude_enabled: bool
    exclude_path_patterns: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class Settings:
    docs_root: Path
    data_dir: Path
    ollama: OllamaCfg
    embedding: EmbeddingCfg
    llamacpp_embed: LlamaCppEmbedCfg
    remote_embed: RemoteEmbedCfg
    chunking: ChunkCfg
    ingest: IngestCfg
    query: QueryCfg
    privacy: PrivacyCfg

    # 派生路径
    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def skipped_path(self) -> Path:
        return self.data_dir / "skipped_files.txt"


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key) or {}
    if not isinstance(val, dict):
        raise ValueError(f"配置项 {key} 应为映射，实际为 {type(val).__name__}")
    return val


def _resolve_api_key(settings_value: str) -> str:
    """解析远程服务的 api_key（参见 ~/Projects/agent 的密钥管理）。

    密钥不写入会被提交的配置：config/settings.yaml 里只留空占位，真实值放
    在被 .gitignore 忽略的 user/secrets.yaml，或用环境变量注入（CI/临时用）。

    优先级（与 agent 的 providers/xf-yun.js 一致：本地配置优先、环境变量兜底）：
        1. user/secrets.yaml 的 remote_embed.api_key（本地明文，user/ 已忽略）
        2. 环境变量 XF_YUN_API_KEY
        3. settings.yaml 里的值（通常为空占位）
    """
    secrets_file = PROJECT_ROOT / "user" / "secrets.yaml"
    if secrets_file.exists():
        try:
            with secrets_file.open("r", encoding="utf-8") as f:
                sec = yaml.safe_load(f) or {}
            key = (_section(sec, "remote_embed") or {}).get("api_key")
            if key:
                return str(key)
        except (OSError, yaml.YAMLError, ValueError):
            pass  # 密钥文件损坏/不可读时静默退化到环境变量
    return str(os.environ.get("XF_YUN_API_KEY") or settings_value or "")


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """读取配置。允许用环境变量 RAGDOCS_CONFIG 覆盖默认路径。"""
    cfg_path = Path(path or os.environ.get("RAGDOCS_CONFIG") or DEFAULT_CONFIG)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    o = _section(raw, "ollama")
    c = _section(raw, "chunking")
    i = _section(raw, "ingest")
    q = _section(raw, "query")
    p = _section(raw, "privacy")
    e = _section(raw, "embedding")
    le = _section(raw, "llamacpp_embed")
    re_ = _section(raw, "remote_embed")

    return Settings(
        docs_root=Path(raw["docs_root"]).expanduser(),
        data_dir=Path(raw["data_dir"]).expanduser(),
        ollama=OllamaCfg(
            base_url=o.get("base_url", "http://127.0.0.1:11434"),
            embed_model=o.get("embed_model", "bge-m3"),
            llm_model=o.get("llm_model", "qwen2.5:7b"),
            request_timeout=float(o.get("request_timeout", 120)),
            restart_cmd=o.get("restart_cmd", "sudo systemctl restart ollama"),
        ),
        embedding=EmbeddingCfg(
            backend=str(e.get("backend", "ollama")),
            local_model=str(e.get("local_model", "BAAI/bge-m3")),
        ),
        llamacpp_embed=LlamaCppEmbedCfg(
            model_path=str(le.get("model_path", "~/models/bge-m3-q8_0.gguf")),
            n_ctx=int(le.get("n_ctx", 8192)),
            n_threads=int(le.get("n_threads", 0)),
            n_gpu_layers=int(le.get("n_gpu_layers", 0)),
            batch_size=int(le.get("batch_size", 32)),
        ),
        remote_embed=RemoteEmbedCfg(
            base_url=str(re_.get("base_url", "")),
            api_key=_resolve_api_key(str(re_.get("api_key", ""))),
            model=str(re_.get("model", "")),
            timeout=float(re_.get("timeout", 60.0)),
            dims=int(re_.get("dims", 0)),
            batch_size=int(re_.get("batch_size", 8)),
        ),
        chunking=ChunkCfg(
            chunk_size=int(c.get("chunk_size", 512)),
            chunk_overlap=int(c.get("chunk_overlap", 64)),
            short_doc_threshold=int(c.get("short_doc_threshold", 1200)),
            inject_fields=tuple(c.get("inject_fields", [])),
        ),
        ingest=IngestCfg(
            include_ext=frozenset(str(x).lower() for x in i.get("include_ext", [])),
            exclude_dirs=frozenset(str(x) for x in i.get("exclude_dirs", [])),
            max_file_bytes=int(i.get("max_file_bytes", 524288)),
        ),
        query=QueryCfg(
            top_k=int(q.get("top_k", 5)),
            similarity_threshold=float(q.get("similarity_threshold", 0.0)),
            system_prompt=str(q.get("system_prompt", "")).strip(),
        ),
        privacy=PrivacyCfg(
            exclude_enabled=bool(p.get("exclude_enabled", False)),
            exclude_path_patterns=tuple(p.get("exclude_path_patterns", [])),
            mode=str(p.get("mode", "skip")),
        ),
    )
