#!/usr/bin/env python3
"""命令行检索/问答，用于不挂 agent 时直接验证效果。

用法：
    .venv/bin/python scripts/query.py search "transformer 的注意力机制"
    .venv/bin/python scripts/query.py ask "我的笔记里怎么理解控制论？"
    .venv/bin/python scripts/query.py stats
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragdocs.config import load_settings
from ragdocs.core.index import RagIndex


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", "--top-k", type=int, default=5)
    s.add_argument("-t", "--threshold", type=float, default=0.0)

    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("-k", "--top-k", type=int, default=5)

    sub.add_parser("stats")

    args = ap.parse_args()
    rag = RagIndex(load_settings())

    if args.cmd == "stats":
        import json
        print(json.dumps(rag.stats(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "search":
        hits = rag.search(args.query, top_k=args.top_k, threshold=args.threshold)
        if not hits:
            print("知识库中未找到相关内容。")
            return 0
        for i, h in enumerate(hits, 1):
            score = f"{h.score:.4f}" if h.score is not None else "-"
            print(f"\n[{i}] {h.path}  ({h.topic}, score={score})")
            print("-" * 60)
            print(h.text[:600])
        return 0

    if args.cmd == "ask":
        answer, hits = rag.ask(args.question, top_k=args.top_k)
        print(answer)
        print("\n引用:")
        for h in hits:
            print(f"  - {h.path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
