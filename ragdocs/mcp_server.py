"""MCP Server：把 RAG 能力暴露成 agent 可调用的工具。

协议说明（学习资料会展开）：
  MCP (Model Context Protocol) 是 agent 调外部工具的事实标准。
  stdio 传输下，agent 把本脚本当子进程拉起，双方通过 stdin/stdout 交换
  JSON-RPC 消息。好处：不用开端口、不用管鉴权、随 agent 生命周期启停。

关于 SDK 版本：本机装到的是 mcp 2.3.0。2.x 把 1.x 的 FastMCP 改名为
MCPServer，并推荐用 `@server.tool()` 直接装饰普通函数——工具名、描述、
输入 schema 全部从函数名 / docstring / 类型注解自动生成，比 1.x 手写
`list_tools` + `call_tool` 两个大分派函数干净得多。本项目按 2.x 写。

⚠️ stdio 传输的头号坑：stdout 是 JSON-RPC 专用通道，往 stdout 打任何
调试信息都会破坏协议。所有日志必须走 stderr（见 _log）。

暴露 4 个工具：
    search_docs   纯检索，返回带出处的片段（省 LLM token，最常用）
    ask_docs      检索 + 本地 LLM 生成答案，附引用来源
    reindex_docs  增量重建索引
    rag_stats     索引规模 / 模型 / 主题分布 / 隐私开关状态
"""
from __future__ import annotations

import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from .config import load_settings
from .core.index import RagIndex


def _log(msg: str) -> None:
    """只写 stderr。stdout 属于 MCP 协议，不能碰。"""
    print(f"[ragdocs] {msg}", file=sys.stderr, flush=True)


def _hit_dict(h) -> dict[str, Any]:
    return {
        "path": h.path,
        "title": h.title,
        "topic": h.topic,
        "score": round(h.score, 4) if h.score is not None else None,
        "text": h.text,
    }


def create_server() -> MCPServer:
    cfg = load_settings()
    rag = RagIndex(cfg)
    server = MCPServer(
        "ragdocs",
        instructions="个人知识库检索工具集，文档来自 ~/Documents 的 Markdown/代码笔记。",
    )

    @server.tool()
    def search_docs(query: str, top_k: int = 5, threshold: float = 0.0) -> dict:
        """在个人知识库中做语义检索，返回最相关原文片段及文件路径。

        适合查找概念定义、笔记原文、某领域资料。只检索、不调用 LLM，
        速度快且不消耗生成额度。query 支持中文。

        Args:
            query: 检索问题或关键词
            top_k: 返回片段数，默认 5
            threshold: 最低相似度 0~1，0 表示不过滤
        """
        hits = rag.search(query, top_k=top_k, threshold=threshold)
        return {"count": len(hits), "hits": [_hit_dict(h) for h in hits]}

    @server.tool()
    def ask_docs(question: str, top_k: int = 5) -> dict:
        """基于个人知识库问答：先检索片段，再让本地 LLM 依据片段生成简体中文回答。

        需要综合多篇笔记时用这个；只要原文请用 search_docs。
        返回 answer 与 sources（引用文件路径列表），便于溯源。
        """
        answer, hits = rag.ask(question, top_k=top_k)
        return {
            "answer": answer,
            "sources": [
                {"path": h.path, "title": h.title,
                 "score": round(h.score, 4) if h.score is not None else None}
                for h in hits
            ],
        }

    @server.tool()
    def reindex_docs() -> dict:
        """增量重建向量索引，只处理新增/修改/删除的文件，未改动文件跳过。

        刚写入或修改过 ~/Documents 里的笔记后调用。返回本次构建统计。
        """
        _log("开始增量重建索引…")
        return rag.build().__dict__

    @server.tool()
    def rag_stats() -> dict:
        """查看索引状态：文件数、chunk 数、按主题分布、模型名、隐私开关。用于排障。"""
        return rag.stats()

    return server


def main() -> None:
    _log("MCP server 启动（stdio）")
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
