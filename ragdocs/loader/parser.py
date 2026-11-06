"""文档解析与分块：把磁盘上的文件变成带元数据的 TextNode 原料。

这一层是整个 RAG 里最「个性化」的部分，也是效果差异最大的地方。
通用教程会直接 SimpleDirectoryReader + SentenceSplitter 一把梭，
但你的库里：
  1. 每个 md 都有 YAML frontmatter（title/status/updated）——是天然的元数据；
  2. 大量 [[双链]] 是 Obsidian 风格的内部引用，直接当普通文本会损失语义；
  3. 很多笔记极短（model.md 只有十来行），硬切 512 会把一条完整概念劈成两半。

因此这里做三件事：
  - 解析 frontmatter，抽出 title/status/updated，并从一级目录推导 topic；
  - 把 [[双链]] 的方括号去掉但保留词，避免检索时「[[小模型]]」和「小模型」对不上；
  - 短文档整篇不切，长文档按字符窗口切（overlap 保证跨窗语义连续）。

产出的是纯 dataclass（Chunk），不含 llama-index 对象。
理由：加载层不该依赖向量框架，将来换框架（LangChain / 自研）时这层能原样复用。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import Settings

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
# [[目标]] 或 [[目标|显示名]]：统一取「目标」，因为库内链接多用目标名作为概念词
_RAGLINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")


@dataclass(frozen=True)
class DocMeta:
    """从路径与 frontmatter 得到的文档级元数据。"""
    path: str                 # 相对 docs_root 的路径
    abs_path: str
    title: str
    status: str
    updated: str
    topic: str                # 一级目录名，如 "2.6-intelligence"
    ext: str
    frontmatter: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """一个待向量化的文本单元。"""
    doc_id: str               # 由相对路径哈希得到，稳定且与绝对路径解耦
    chunk_id: str             # doc_id + 序号
    text: str                 # 注入过元数据的最终文本（拿去 embedding 的就是它）
    raw_text: str             # 未注入元数据的原文（查询时展示给用户看）
    meta: DocMeta
    ordinal: int


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """返回 (frontmatter 字典, 去掉 frontmatter 的正文)。失败时返回空字典与原文。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
        if not isinstance(data, dict):
            data = {}
    except yaml.YAMLError:
        data = {}
    return data, text[m.end():]


def _topic_of(rel: str) -> str:
    """一级目录名。你的库用 0.x / 1.x / 2.x 编号，本身就是主题标签。"""
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else "(root)"


def build_meta(path: Path, cfg: Settings, text: str) -> DocMeta:
    rel = str(path.relative_to(cfg.docs_root))
    fm, _ = _parse_frontmatter(text)
    return DocMeta(
        path=rel,
        abs_path=str(path),
        title=str(fm.get("title") or path.stem),
        status=str(fm.get("status") or ""),
        updated=str(fm.get("updated") or ""),
        topic=_topic_of(rel),
        ext=path.suffix.lower(),
        frontmatter=fm,
    )


def _normalize_raglinks(text: str) -> str:
    """[[小模型]] -> 小模型；[[目标|显示名]] -> 目标。"""
    return _RAGLINK_RE.sub(lambda m: m.group(1).strip(), text)


def _make_header(meta: DocMeta, cfg: Settings) -> str:
    """拼 chunk 头部。只注入配置里列出的字段，保持可控。

    为什么把元数据「写进 embedding 文本」而不是只放 metadata 字段：
    bge-m3 这类模型只对 text 编码。用户问「2.6-intelligence 里关于
    transformer 的笔记」时，若 topic 没进文本，纯向量检索几乎命中不了
    「哪个目录」这个条件。写进文本是最省事、最有效的元数据融合方式。
    """
    available = {
        "path": meta.path,
        "title": meta.title,
        "topic": meta.topic,
        "status": meta.status,
        "updated": meta.updated,
        "ext": meta.ext,
    }
    parts = [f"{k}: {available[k]}" for k in cfg.chunking.inject_fields
             if k in available and available[k]]
    return "[" + " | ".join(parts) + "]\n" if parts else ""


def _split_long(text: str, cfg: Settings) -> list[str]:
    """按字符窗口切分。用 chunk_size*4 近似 token 数（中英混排下的经验值）。

    优先在换行/句号处收尾，避免把一行代码或一个词从中间劈开。
    """
    size = max(64, cfg.chunking.chunk_size * 4)
    overlap = min(cfg.chunking.chunk_overlap * 4, size // 2)
    n = len(text)
    if n <= size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # 在窗口后 1/4 区间里找最后一个换行或中文句号作为切点
            search_from = start + int(size * 0.75)
            window = text[search_from:end]
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind(". "))
            if cut != -1:
                end = search_from + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def parse_file(path: Path, cfg: Settings) -> list[Chunk]:
    """读一个文件，返回若干 Chunk。编码问题按「尽量读、不崩」处理。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="gbk")
        except (UnicodeDecodeError, OSError):
            return []          # 二进制或异常编码，交给 discovery 之外的兜底跳过
    except OSError:
        return []

    meta = build_meta(path, cfg, raw)
    fm, body = _parse_frontmatter(raw)
    _ = fm  # 已在 build_meta 中解析，这里只取正文
    body = _normalize_raglinks(body).strip()
    if not body:
        return []

    doc_id = hashlib.sha1(meta.path.encode("utf-8")).hexdigest()[:16]
    header = _make_header(meta, cfg)

    # 短文档整篇一个 chunk；长文档才切
    if len(body) <= cfg.chunking.short_doc_threshold:
        pieces = [body]
    else:
        pieces = _split_long(body, cfg)

    out: list[Chunk] = []
    for i, piece in enumerate(pieces):
        out.append(Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}:{i}",
            text=header + piece,
            raw_text=piece,
            meta=meta,
            ordinal=i,
        ))
    return out
