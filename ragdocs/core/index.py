"""向量索引：把 Chunk 变成可检索的 VectorStoreIndex，并支持落盘/载入/增量更新。

llamaIndex 组件映射（这是学习资料里会反复对照的表）：

    本项目概念            llamaIndex 对象                 作用
    -------------------  -----------------------------  ---------------------------
    Chunk                TextNode(id_=chunk_id, text=)  最小检索单元，带 metadata
    Ollama bge-m3        OllamaEmbedding                 text -> 向量
    SimpleVectorStore    SimpleVectorStore              内存向量表，可 persist 成 json
    StorageContext       StorageContext                 把 vector store / docstore 管起来
    VectorStoreIndex     VectorStoreIndex               对外提供 retriever / query
    Ollama qwen          Ollama(llm)                    ask 时做带上下文的生成

关于「为什么不用 chromadb」：
  1. 它在 Python 3.14 上需要现场编译，装依赖这一步就已经卡过一次；
  2. 2208 篇笔记、假设每篇 1~3 个 chunk，总量约 5000 条向量，
     SimpleVectorStore（纯 Python dict + 余弦相似度）完全够用；
  3. 少一个服务进程，MCP server 启动更快、更不容易挂。
  真到了十万级 chunk 再换 Qdrant/Milvus，换的只是 StorageContext 一行。

关于「延迟导入」：
  把 `from llama_index...` 放进函数体，而不是模块顶层。这样本项目里
  「配置/发现/解析/清单」四层可以在没装 llama-index、没起 Ollama 的机器上
  直接跑单测或做讲解；只有真正要建索引/查询时才需要重依赖。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..loader.discovery import discover
from ..loader.manifest import commit_manifest, compute_delta, load_manifest
from ..loader.parser import Chunk, build_meta, parse_file


@dataclass(frozen=True)
class IngestStats:
    total_files: int
    added: int
    changed: int
    removed: int
    unchanged: int
    moved: int
    chunks_built: int
    skipped: int
    total_chunks: int


@dataclass(frozen=True)
class Hit:
    """一条检索结果，给 MCP 工具 / CLI 用，不暴露 llama-index 对象。"""
    chunk_id: str
    doc_id: str
    path: str
    title: str
    topic: str
    score: float | None
    text: str            # 原文片段（不含注入的头部），用于展示


def _build_service_context(cfg: Settings):
    """构造 llamaIndex 的全局 Settings 等价物（embed_model + llm）。

    0.14 版推荐用 Settings.embed_model / Settings.llm 全局注入，
    比老教程里到处传 service_context 干净。

    embed_model 的选择取决于 embedding 后端：
    - "ollama" / "local"：用 OllamaEmbedding（query 侧仍需一个 embed 模型；
      local 后端虽有 sentence-transformers 做批量 embed，但 query 实时向量化
      仍走这里）。
    - "llamacpp"：用 `_LlamaCppEmbeddingAdapter` 包裹进程内的 llama.cpp 模型，
      这样 search() 的 retriever 把 query 向量化时不再依赖 Ollama 服务。
    - "remote"：用 `_RemoteEmbeddingAdapter` 把 query 向量化转发到远程
      embedding 服务（OpenAI /embeddings 协议），适合本机无 GPU 的场景。
    """
    from llama_index.core import Settings as LISettings
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.llms.ollama import Ollama

    if cfg.embedding.backend == "llamacpp":
        # 局部定义适配器，继承函数内导入的 BaseEmbedding，保持模块顶层零重依赖。
        # 供 retriever 把 query 向量化——search() 不再依赖 Ollama 服务。
        class _LlamaCppEmbeddingAdapter(BaseEmbedding):
            def __init__(self):
                super().__init__(
                    model_name=f"llamacpp:{Path(cfg.llamacpp_embed.model_path).name}"
                )

            def _embed_one(self, text: str) -> list[float]:
                import numpy as np

                model = _get_llamacpp_model(cfg)
                vec = np.asarray(model.embed(text), dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()

            def _get_query_embedding(self, query: str) -> list[float]:
                return self._embed_one(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._embed_one(query)

            def _get_text_embedding(self, text: str) -> list[float]:
                return self._embed_one(text)

        embed = _LlamaCppEmbeddingAdapter()
    elif cfg.embedding.backend == "remote":
        # 局部定义适配器，query 向量化走远程 embedding 服务（与批量 _embed_remote 同源）。
        # 注意：BaseEmbedding 是 pydantic 模型，动态设置未声明字段（如 self._cfg）
        # 会被忽略，故这里用闭包捕获 cfg，方法内直接用，不存 self。
        class _RemoteEmbeddingAdapter(BaseEmbedding):
            def __init__(self):
                super().__init__(model_name=f"remote:{cfg.remote_embed.model}")

            def _embed_one(self, text: str) -> list[float]:
                return _embed_remote([text], cfg, batch_size=1)[0]

            def _get_query_embedding(self, query: str) -> list[float]:
                return self._embed_one(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._embed_one(query)

            def _get_text_embedding(self, text: str) -> list[float]:
                return self._embed_one(text)

        embed = _RemoteEmbeddingAdapter()
    else:
        from llama_index.embeddings.ollama import OllamaEmbedding

        embed = OllamaEmbedding(
            model_name=cfg.ollama.embed_model,
            base_url=cfg.ollama.base_url,
            # 0.14 的 OllamaEmbedding 没有 request_timeout 形参，超时要经
            # client_kwargs 传给底层 ollama.Client；LLM 侧则直接支持该参数。
            client_kwargs={"timeout": cfg.ollama.request_timeout},
        )
    llm = Ollama(
        model=cfg.ollama.llm_model,
        base_url=cfg.ollama.base_url,
        request_timeout=cfg.ollama.request_timeout,
    )
    LISettings.embed_model = embed
    LISettings.llm = llm
    return embed, llm


def _chunk_to_node(chunk: Chunk):
    from llama_index.core.schema import TextNode

    return TextNode(
        id_=chunk.chunk_id,
        text=chunk.text,                 # 注入了元数据的文本，用于 embedding
        # ref_doc_id 是 SimpleVectorStore 删除操作的粒度：delete(ref_doc_id)
        # 会清掉该值下的所有 node。把它设成 doc_id，就能「按文件」精确删除，
        # 不必去碰 store.data.embedding_dict 这种内部结构。
        ref_doc_id=chunk.doc_id,
        metadata={
            "doc_id": chunk.doc_id,
            "path": chunk.meta.path,
            "title": chunk.meta.title,
            "topic": chunk.meta.topic,
            "status": chunk.meta.status,
            "updated": chunk.meta.updated,
            "ext": chunk.meta.ext,
            "ordinal": chunk.ordinal,
            "raw_text": chunk.raw_text,  # 冗余存一份，查询时直接展示，不必回读文件
        },
        # 让 LLM 生成时只看「干净」的正文，不重复看头部元数据
        excluded_embed_metadata_keys=["raw_text", "ordinal", "doc_id", "ext", "updated", "status"],
        excluded_llm_metadata_keys=["raw_text", "ordinal", "doc_id", "ext"],
    )


def _node_to_hit(node, score: float | None) -> Hit:
    md = node.metadata or {}
    return Hit(
        chunk_id=node.node_id,
        doc_id=str(md.get("doc_id", "")),
        path=str(md.get("path", "")),
        title=str(md.get("title", "")),
        topic=str(md.get("topic", "")),
        score=score,
        text=str(md.get("raw_text") or node.text),
    )


def _batch_embed(texts: list[str], cfg: Settings, batch_size: int = 64) -> list[list[float]]:
    """批量把文本变成向量。根据 `cfg.embedding.backend` 选择后端：

    - "ollama"：直接调 Ollama `/api/embed`（多文本一次请求）。
      本机实测 bge-m3 worker 易卡死，故带「超时即重启 Ollama 重试」自愈。
    - "local"：进程内用 sentence-transformers 跑模型，无服务依赖、稳定。

    不论哪个后端，都返回与 texts 等长的向量列表。节点预填 embedding 后
    交给 VectorStoreIndex，llamaIndex 便不会重算。

    **为什么不用 llamaIndex 的 OllamaEmbedding 来 embed 节点**：
    实测它对每个节点**逐条**调 Ollama，60 个 chunk 要 55 秒；直接批量调
    /api/embed 同样 60 条 1.8 秒，差 30 倍。故本项目自己批量 embed。
    """
    if cfg.embedding.backend == "local":
        return _embed_local(texts, cfg, batch_size)
    if cfg.embedding.backend == "llamacpp":
        return _embed_llamacpp(texts, cfg, batch_size)
    if cfg.embedding.backend == "remote":
        # 远程服务单请求宜小批量，用专属 batch_size（避免 64 条长文本一次性
        # POST 导致服务端处理超时，见 docs/06 第 7 节）
        return _embed_remote(texts, cfg, cfg.remote_embed.batch_size)
    return _embed_ollama(texts, cfg, batch_size)


def _embed_ollama(texts: list[str], cfg: Settings, batch_size: int = 64,
                  _retried: bool = False) -> list[list[float]]:
    import json
    import subprocess
    import sys
    import time
    import urllib.request
    from urllib.error import URLError

    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps(
            {"model": cfg.ollama.embed_model, "input": batch}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{cfg.ollama.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.ollama.request_timeout) as resp:
                data = json.loads(resp.read())
        except (TimeoutError, URLError, ConnectionError) as e:
            if _retried or not cfg.ollama.restart_cmd:
                raise
            # 卡死：重启 Ollama 清掉坏 worker，等重新监听后重试整批
            print(f"[ragdocs] embed 卡死({type(e).__name__})，重启 Ollama 后重试…",
                  file=sys.stderr, flush=True)
            try:
                subprocess.run(cfg.ollama.restart_cmd, shell=True, check=True,
                               timeout=30, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except Exception:
                pass
            time.sleep(4)
            return _embed_ollama(texts, cfg, batch_size, _retried=True)
        out.extend(data["embeddings"])
    return out


_local_model = None  # 进程内缓存，避免重复加载


def _embed_local(texts: list[str], cfg: Settings, batch_size: int = 32) -> list[list[float]]:
    """进程内用 sentence-transformers 跑 embedding，CPU 即可、稳定。"""
    import numpy as np

    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[ragdocs] 加载本地嵌入模型 {cfg.embedding.local_model} …",
              file=sys.stderr, flush=True)
        _local_model = SentenceTransformer(cfg.embedding.local_model)

    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = _local_model.encode(
            batch, normalize_embeddings=True, show_progress_bar=False
        )
        out.extend(np.asarray(vecs).tolist())
    return out


# ---------------------------------------------------------------
# llama.cpp (GGUF) embedding 后端
# ---------------------------------------------------------------
# llama.cpp 直接读本地 GGUF 文件、进程内推理，无 Ollama 服务依赖。
# 与 _local_model 同模式，进程内缓存模型句柄避免重复加载。

_llamacpp_embed_model = None


def _get_llamacpp_model(cfg: Settings):
    """懒加载并缓存 llama.cpp 的 embedding 模型句柄。"""
    global _llamacpp_embed_model
    if _llamacpp_embed_model is None:
        from llama_cpp import Llama

        model_path = Path(cfg.llamacpp_embed.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                f"llama.cpp 嵌入模型不存在: {model_path}\n"
                f"请先下载 GGUF 版 bge-m3（如 CompendiumLabs/bge-m3-gguf）并填入 "
                f"config/settings.yaml 的 llamacpp_embed.model_path。"
            )
        print(f"[ragdocs] 加载 llama.cpp 嵌入模型 {model_path.name} …",
              file=sys.stderr, flush=True)
        _llamacpp_embed_model = Llama(
            model_path=str(model_path),
            embedding=True,                 # 开启 embedding 模式（不初始化 KV 缓存做生成）
            n_ctx=cfg.llamacpp_embed.n_ctx,
            n_threads=cfg.llamacpp_embed.n_threads or 0,
            n_gpu_layers=cfg.llamacpp_embed.n_gpu_layers,
            verbose=False,
        )
    return _llamacpp_embed_model


def _embed_llamacpp(texts: list[str], cfg: Settings, batch_size: int = 32) -> list[list[float]]:
    """用 llama.cpp 进程内批量 embedding，无服务依赖、稳定。

    llama.cpp 的 `Llama.embed()` 接受单条文本或文本列表；这里按 batch_size
    切块循环，避免一次塞入过多文本撑爆上下文。bge-m3 输出默认已 L2 归一化，
    但为稳妥仍对结果再做一次归一化，与 sentence-transformers 的
    normalize_embeddings=True 行为对齐。
    """
    import numpy as np

    if not texts:
        return []
    model = _get_llamacpp_model(cfg)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # embed() 返回 shape (n, dim) 的 numpy 数组（单条时也可能为 1-D）
        vecs = np.asarray(model.embed(batch), dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        # L2 归一化，保证与索引构建时一致的相似度计算
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out.extend((vecs / norms).tolist())
    return out


# ---------------------------------------------------------------
# 远程 embedding 服务（OpenAI 兼容 /embeddings 协议，如讯飞 MaaS）
# ---------------------------------------------------------------

def _embed_remote(texts: list[str], cfg: Settings, batch_size: int = 32) -> list[list[float]]:
    """把文本批量发到远程 embedding 服务（默认 OpenAI /v1/embeddings 协议）。

    适用于本机无可用 GPU、CPU 推理过慢的场景。注意：
    - 文本会出网；
    - 远程模型维度（如 768）通常与本地 bge-m3（1024）不同，切换后需删
      data/ 全量重建；
    - 接口支持 input 传数组，这里按 batch_size 切块一次多文本，减少往返。
    """
    import json
    import time
    import urllib.request
    from urllib.error import URLError

    # 网络层偶发错误：超时、连接重置、对端断开都该重试，而非直接崩掉整个构建
    _RETRYABLE = (TimeoutError, URLError, ConnectionError, ConnectionResetError, OSError)

    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({
            "model": cfg.remote_embed.model,
            "input": batch,
            **({"dimensions": cfg.remote_embed.dims} if cfg.remote_embed.dims else {}),
        }).encode("utf-8")
        last_err: Exception | None = None
        data = None
        for attempt in range(3):  # 最多重试 2 次（共 3 次尝试）
            req = urllib.request.Request(
                cfg.remote_embed.base_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {cfg.remote_embed.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=cfg.remote_embed.timeout) as resp:
                    data = json.loads(resp.read())
                break
            except _RETRYABLE as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s / 2s 退避
                    continue
        if data is None:
            raise RuntimeError(
                f"远程 embedding 调用失败（重试耗尽，末次 {type(last_err).__name__}）：{last_err}"
            ) from last_err
        # data["data"] 顺序与 input 一致，按 index 兜底排序以防服务端乱序
        items = {it.get("index", k): it["embedding"] for k, it in enumerate(data["data"])}
        out.extend(items[k] for k in sorted(items))
    return out


class RagIndex:
    """索引的门面。所有对外的 build / query / ask / stats 都走这里。

    生命周期：
        idx = RagIndex(cfg)
        idx.build()                 # 首次全量 / 之后增量
        idx.load()                  # 从磁盘恢复（不重新 embedding）
        hits = idx.search("...")
        answer = idx.ask("...")
    """

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._index = None

    # ---------- 构建 ----------

    def build(self, write_manifest: bool = True) -> IngestStats:
        """增量构建。返回统计信息。

        流程：发现文件 -> 对比 manifest 得 Delta -> 只对 added/changed 解析分块
        -> 已有索引则 insert/delete，否则新建 -> persist -> (可选)提交 manifest。

        write_manifest=False 用于「小批量先验证」：建索引、能检索，但不污染
        全量增量状态（否则全量时会以为这些文件已索引而跳过）。
        """
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.vector_stores import SimpleVectorStore

        disc = discover(self.cfg)
        delta, stamps = compute_delta(self.cfg, disc.files)
        to_process = list(delta.added) + list(delta.changed)

        new_nodes = []
        chunk_counts: dict[str, int] = {}
        for p in to_process:
            chunks = parse_file(p, self.cfg)
            rel = str(p.relative_to(self.cfg.docs_root))
            chunk_counts[rel] = len(chunks)
            new_nodes.extend(_chunk_to_node(c) for c in chunks)

        _build_service_context(self.cfg)   # 注入 embed_model（供 query 侧 embed query）

        # 自己批量预填 embedding（绕过 llamaIndex 逐条调用的 30 倍慢路径）
        if new_nodes:
            texts = [n.text for n in new_nodes]
            vecs = _batch_embed(texts, self.cfg)
            for n, v in zip(new_nodes, vecs):
                n.embedding = v

        if self._index is None and self.cfg.index_dir.exists():
            self.load()

        if self._index is None:
            if not new_nodes:
                # 空库：建一个空索引，保证后续 insert 可用
                store = SimpleVectorStore()
                ctx = StorageContext.from_defaults(vector_store=store)
                self._index = VectorStoreIndex([], storage_context=ctx)
            else:
                store = SimpleVectorStore()
                ctx = StorageContext.from_defaults(vector_store=store)
                self._index = VectorStoreIndex(new_nodes, storage_context=ctx)
        else:
            # 删除已移除/被修改文件的旧节点，再插入新节点
            stale_ids = set()
            for rel in delta.removed:
                stale_ids.add(rel)
            for p in delta.changed:
                stale_ids.add(str(p.relative_to(self.cfg.docs_root)))
            if stale_ids:
                self._delete_by_paths(stale_ids)
            if new_nodes:
                self._index.insert_nodes(new_nodes)

        # 改名/移动：内容未变，向量完全复用，只就地重贴路径元数据（不重 embed）
        self._relabel_moved(delta.moved)

        self._persist()
        if write_manifest:
            commit_manifest(self.cfg, stamps, chunk_counts)

        total_chunks = sum(
            int(rec.get("chunks", 0))
            for rec in load_manifest(self.cfg)["files"].values()
        )
        return IngestStats(
            total_files=len(disc.files),
            added=len(delta.added),
            changed=len(delta.changed),
            removed=len(delta.removed),
            unchanged=len(delta.unchanged),
            moved=len(delta.moved),
            chunks_built=len(new_nodes),
            skipped=len(disc.skipped),
            total_chunks=total_chunks,
        )

    def _delete_by_paths(self, rel_paths: set[str]) -> None:
        """按文件路径删除其所有 chunk。

        走官方 API：SimpleVectorStore.delete(ref_doc_id) 会清掉
        text_id_to_ref_doc_id 中该 ref_doc_id 对应的全部 node。
        因为 _chunk_to_node 把 ref_doc_id 设成了 doc_id（= sha1(路径)[:16]），
        这里反算出 doc_id 直接删即可，一个文件一次调用。
        """
        import hashlib

        store = self._index.storage_context.vector_store
        for rel in rel_paths:
            doc_id = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
            store.delete(doc_id)

    def _relabel_moved(self, moved: tuple[tuple[str, str], ...]) -> None:
        """把改名/移动的文件在索引里就地重贴路径元数据，不重 embed、不重建。

        compute_delta 已用内容 sha1 把 removed+added 还原成 mv：内容没变，
        向量完全复用。这里只做两件事：
          1. 用新路径重新派生 path/topic/title（读一次本地文件，不 parse 成 chunk、
             不 embed）；
          2. 找到 docstore 里 metadata.path==旧路径 的节点，就地改元数据并
             add_documents(allow_update=True) 覆盖；同步 vector_store 的冗余
             metadata_dict，保持一致。

        刻意不动：embedding_dict（向量）、text（内嵌旧 header 的 embedding 文本）、
        doc_id / chunk_id / ref_doc_id（冻结为旧路径哈希，改了会破坏向量键）。
        这是「不重建」的代价，详见 docs/03。
        """
        if not moved:
            return
        index = self._ensure()
        ctx = index.storage_context
        docstore = ctx.docstore
        metadata_dict = ctx.vector_store.data.metadata_dict

        for old_rel, new_rel in moved:
            new_path = self.cfg.docs_root / new_rel
            try:
                text = new_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                text = ""
            new_meta = build_meta(new_path, self.cfg, text)

            updated = []
            for node_id, node in docstore.docs.items():
                if node.metadata.get("path") != old_rel:
                    continue
                md = node.metadata
                md["path"] = new_rel
                md["topic"] = new_meta.topic
                md["title"] = new_meta.title
                if "abs_path" in md:
                    md["abs_path"] = str(new_path)
                updated.append(node)
                # 同步 vector_store 里的冗余副本（retrieve 实际读 docstore，
                # 这份仅用于一致性与审计，取不到就跳过不影响正确性）
                m = metadata_dict.get(node_id)
                if m is not None:
                    m["path"] = new_rel
                    m["topic"] = new_meta.topic
                    m["title"] = new_meta.title

            if updated:
                docstore.add_documents(updated, allow_update=True)

    def _persist(self) -> None:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        self._index.storage_context.persist(persist_dir=str(self.cfg.index_dir))
        # 顺手把跳过清单写一份，方便你审计「哪些文件没进索引」
        disc = discover(self.cfg)
        lines = [f"{s.reason}\t{s.path}" for s in disc.skipped]
        self.cfg.skipped_path.write_text("\n".join(lines), encoding="utf-8")

    # ---------- 载入 ----------

    def load(self) -> "RagIndex":
        """从磁盘恢复索引。不调用 embedding，启动极快。"""
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.core.vector_stores import SimpleVectorStore

        _build_service_context(self.cfg)
        if not self.cfg.index_dir.exists():
            raise FileNotFoundError(
                f"索引尚不存在: {self.cfg.index_dir}。请先运行 scripts/build_index.py"
            )
        ctx = StorageContext.from_defaults(
            persist_dir=str(self.cfg.index_dir),
            vector_store=SimpleVectorStore.from_persist_dir(str(self.cfg.index_dir)),
        )
        self._index = load_index_from_storage(ctx)
        return self

    def _ensure(self):
        if self._index is None:
            self.load()
        return self._index

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int | None = None,
               threshold: float | None = None) -> list[Hit]:
        """纯向量检索，不调用 LLM。适合 agent 自己决定怎么用片段。"""
        k = top_k or self.cfg.query.top_k
        th = self.cfg.query.similarity_threshold if threshold is None else threshold
        index = self._ensure()
        retriever = index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        hits = []
        for n in nodes:
            score = getattr(n, "score", None)
            if score is not None and th > 0 and score < th:
                continue
            hits.append(_node_to_hit(n.node, float(score) if score is not None else None))
        return hits

    def ask(self, question: str, top_k: int | None = None) -> tuple[str, list[Hit]]:
        """检索 + LLM 生成。返回 (回答, 引用片段列表)，便于溯源。

        LLM 是软依赖：若生成模型未就绪（没 pull / Ollama 无该模型），
        不崩溃，返回可读提示，并附上已检索到的片段，让 agent 能退化成
        「只取片段自己组织」。
        """
        hits = self.search(question, top_k=top_k)
        if not hits:
            return "知识库中未找到相关内容。", []

        context = "\n\n".join(
            f"--- 来源: {h.path} (score={h.score:.3f}) ---\n{h.text}" if h.score is not None
            else f"--- 来源: {h.path} ---\n{h.text}"
            for h in hits
        )

        llm = self._build_llm()
        if llm is None:
            return (
                f"检索到 {len(hits)} 个相关片段，但本服务的生成模型 "
                f"「{self.cfg.ollama.llm_model}」未就绪（可能未 `ollama pull`）。"
                f"请改用 search_docs 获取原文片段，或先拉取生成模型。",
                hits,
            )

        prompt = (
            f"{self.cfg.query.system_prompt}\n\n"
            f"=== 检索片段 ===\n{context}\n\n"
            f"=== 问题 ===\n{question}\n\n"
            f"=== 回答 ==="
        )
        resp = llm.complete(prompt)
        return str(resp).strip(), hits

    def _build_llm(self):
        """构造生成模型；若模型在 Ollama 不可用，返回 None（软依赖）。

        探测方式：直接调一次 Ollama 的 list 接口，比「先假设可用再等超时」快。
        """
        from llama_index.llms.ollama import Ollama

        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self.cfg.ollama.base_url}/api/tags", timeout=5
            ) as r:
                import json
                models = {m["name"] for m in json.loads(r.read()).get("models", [])}
            if self.cfg.ollama.llm_model not in models:
                return None
        except Exception:
            # 连不上 Ollama 也视为不可用，交给调用方降级。
            # （不抛异常，否则 search_docs 之外的整条链路会崩。）
            return None

        return Ollama(
            model=self.cfg.ollama.llm_model,
            base_url=self.cfg.ollama.base_url,
            request_timeout=self.cfg.ollama.request_timeout,
        )

    # ---------- 状态 ----------

    def stats(self) -> dict[str, Any]:
        manifest = load_manifest(self.cfg)
        files = manifest["files"]
        by_topic: dict[str, int] = {}
        for rel in files:
            topic = rel.split("/", 1)[0] if "/" in rel else "(root)"
            by_topic[topic] = by_topic.get(topic, 0) + 1
        return {
            "docs_root": str(self.cfg.docs_root),
            "index_dir": str(self.cfg.index_dir),
            "index_exists": self.cfg.index_dir.exists(),
            "files_indexed": len(files),
            "chunks_indexed": sum(int(r.get("chunks", 0)) for r in files.values()),
            "by_topic": dict(sorted(by_topic.items())),
            "embed_model": (
                f"llamacpp:{self.cfg.llamacpp_embed.model_path}"
                if self.cfg.embedding.backend == "llamacpp"
                else f"remote:{self.cfg.remote_embed.model}"
                if self.cfg.embedding.backend == "remote"
                else self.cfg.ollama.embed_model
            ),
            "embed_backend": self.cfg.embedding.backend,
            "llm_model": self.cfg.ollama.llm_model,
            "privacy_exclude_enabled": self.cfg.privacy.exclude_enabled,
        }
