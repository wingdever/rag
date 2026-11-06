"""增量索引清单：记录每个文件上次索引时的状态，避免重复 embedding。

为什么需要它：bge-m3 对 2208 篇笔记做 embedding 是要花时间的（哪怕有 GPU）。
若每次 `reindex` 都全量重算，改一个字也要等全库。manifest 让重建变成
「只处理新增 / 修改 / 删除的文件」。

判定键用三元组而不是只靠 mtime：
  size + mtime_ns + sha1(内容)
- mtime 会被 git checkout、rclone 同步等改动，单独用会误判「变了」；
- 加内容 hash 后，内容没变就不重算，代价只是多读一次文件（本地 IO 很便宜）。

manifest.json 结构：
{
  "version": 1,
  "files": {
    "<相对路径>": {"size": .., "mtime_ns": .., "sha1": .., "chunks": ..}
  }
}
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings


@dataclass(frozen=True)
class FileStamp:
    size: int
    mtime_ns: int
    sha1: str
    chunks: int


@dataclass(frozen=True)
class Delta:
    """一次增量计算的结果。"""
    added: tuple[Path, ...]
    changed: tuple[Path, ...]
    removed: tuple[str, ...]      # 相对路径
    unchanged: tuple[Path, ...]


def _stamp(path: Path, chunk_count: int = 0) -> FileStamp:
    st = path.stat()
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return FileStamp(st.st_size, st.st_mtime_ns, digest.hexdigest(), chunk_count)


def load_manifest(cfg: Settings) -> dict[str, dict]:
    p = cfg.manifest_path
    if not p.exists():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("files"), dict):
            return {"version": 1, "files": {}}
        return data
    except (json.JSONDecodeError, OSError):
        # 清单损坏不该让整个服务起不来：退化成全量重建
        return {"version": 1, "files": {}}


def save_manifest(cfg: Settings, data: dict) -> None:
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cfg.manifest_path)   # 原子替换，避免写一半被 kill 留下坏清单


def compute_delta(cfg: Settings, current: tuple[Path, ...]) -> tuple[Delta, dict[str, FileStamp]]:
    """对比磁盘现状与 manifest，返回 (变更集, 本次实际算出的 stamp 映射)。

    注意：sha1 只对「mtime 或 size 变了 / 新文件」计算，未变文件直接复用
    清单里的旧 sha1，把读取量压到最低。
    """
    old = load_manifest(cfg)["files"]
    rel_of = {str(p.relative_to(cfg.docs_root)): p for p in current}

    added: list[Path] = []
    changed: list[Path] = []
    unchanged: list[Path] = []
    stamps: dict[str, FileStamp] = {}

    for rel, p in rel_of.items():
        rec = old.get(rel)
        st = p.stat()
        if rec is None:
            added.append(p)
            stamps[rel] = _stamp(p)
        elif (rec.get("size") != st.st_size or rec.get("mtime_ns") != st.st_mtime_ns):
            new_stamp = _stamp(p)
            if new_stamp.sha1 == rec.get("sha1"):
                # 时间变了但内容没变（git/rclone 常见）：视为未变，并刷新 stamp
                unchanged.append(p)
                stamps[rel] = FileStamp(new_stamp.size, new_stamp.mtime_ns,
                                        new_stamp.sha1, rec.get("chunks", 0))
            else:
                changed.append(p)
                stamps[rel] = new_stamp
        else:
            unchanged.append(p)
            stamps[rel] = FileStamp(rec["size"], rec["mtime_ns"], rec["sha1"],
                                    rec.get("chunks", 0))

    removed = tuple(rel for rel in old if rel not in rel_of)
    return Delta(tuple(added), tuple(changed), removed, tuple(unchanged)), stamps


def commit_manifest(cfg: Settings, stamps: dict[str, FileStamp],
                    chunk_counts: dict[str, int]) -> dict:
    """把本次索引结果写回清单。chunk_counts 用相对路径索引。"""
    files = {}
    for rel, st in stamps.items():
        files[rel] = {
            "size": st.size,
            "mtime_ns": st.mtime_ns,
            "sha1": st.sha1,
            "chunks": chunk_counts.get(rel, st.chunks),
        }
    data = {"version": 1, "files": files}
    save_manifest(cfg, data)
    return data
