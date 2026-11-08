#!/usr/bin/env python3
"""全量/增量构建索引。

用法：
    .venv/bin/python scripts/build_index.py            # 增量
    .venv/bin/python scripts/build_index.py --dry-run  # 只看会处理哪些文件，不 embedding
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragdocs.config import load_settings
from ragdocs.core.index import RagIndex
from ragdocs.loader.discovery import discover, DiscoveryResult
from ragdocs.loader.delta import compute_delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告变更集，不真正构建")
    ap.add_argument("--only", metavar="SUBDIR",
                    help="只索引某一级子目录，如 '2.6-intelligence' 或 '1.0-lifesystem'。用于先小批量验证。")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="只处理前 N 个待索引文件（按路径排序）。先小批量跑通再全量。")
    args = ap.parse_args()

    cfg = load_settings()
    disc = discover(cfg)

    # 小批量模式标志：过滤后不写 manifest（避免污染全量增量状态），
    # 且把 skipped 也裁到本批范围内，报告才干净。
    partial = bool(args.only or args.limit)
    if args.only:
        base = args.only.strip("/")
        keep = [p for p in disc.files if str(p.relative_to(cfg.docs_root)).startswith(base + "/")]
        dropped = len(disc.files) - len(keep)
        disc = dataclasses.replace(disc, files=tuple(keep))
        print(f"[--only {base}] 仅处理该子目录下 {len(keep)} 个文件（其余 {dropped} 个本次跳过）")
    if args.limit:
        kept = list(disc.files[: args.limit])
        dropped = len(disc.files) - len(kept)
        disc = dataclasses.replace(disc, files=tuple(kept))
        print(f"[--limit {args.limit}] 仅处理前 {len(kept)} 个文件（其余 {dropped} 个本次跳过）")
    if partial:
        # skipped 报告里只保留「本批文件所在范围」的，但其实 skipped 是全局的；
        # 小批量时意义不大，置 0 避免误导。
        disc = dataclasses.replace(disc, skipped=())
    delta, _ = compute_delta(cfg, disc.files)
    print(f"文档根目录 : {cfg.docs_root}")
    print(f"待索引文件 : {len(disc.files)}")
    print(f"新增 {len(delta.added_embeds)} / 修改 {len(delta.matched_embeds) + len(delta.moved_embeds)} / "
          f"删除 {len(delta.left_removed)} / 移动 {len(delta.moved_prims)} / "
          f"未变 {len(delta.matched_prims) + len(delta.moved_prims)}")

    if args.dry_run:
        print("\n--dry-run，未执行 embedding。新增示例:")
        for p in delta.added_embeds[:10]:
            print("  +", p.relative_to(cfg.docs_root))
        return 0

    rag = RagIndex(cfg)
    st = rag.build(write_manifest=not partial, disc=disc)
    print("\n构建完成:")
    for k, v in st.__dict__.items():
        print(f"  {k:16s}: {v}")
    if partial:
        print("\n[小批量模式] 本次未写入 manifest。确认效果后，去掉 --only/--limit 跑全量即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
