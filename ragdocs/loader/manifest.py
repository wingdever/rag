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
    moved: tuple[tuple[str, str], ...] = ()   # (旧路径, 新路径)，内容未变的改名/移动


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


def _detect_moves(
    added_rels: list[str], removed: list[str], old: dict,
    rel_of: dict[str, Path], stamps: dict[str, FileStamp],
) -> list[tuple[str, str]]:
    """把「一批路径消失 + 一批路径出现、且内容相同」还原为改名/移动 (mv)。

    主键仍是相对路径：本函数只是把 added/removed 里内容指纹一致的配对挑出来，
    当作「同一个文件换了路径」，从而免去重 embed。旧 manifest 结构不变（version 1），
    因此是一次兼容迁移，无需删库重建。

    配对键 = (内容 sha1, size)：removed 侧取清单里的旧指纹，added 侧取本轮
    刚算出的 _stamp 指纹。同键下两组按路径排序后 zip 一对一消费，避免重复复用；
    真正「删旧 + 增同内容新文件」也会被判为 mv（无害，省一次 embed）。

    副作用（原地修改入参，避免来回拷贝）：
    - 从 added_rels / removed 中剔除已配对的路径；
    - 把 stamps[new_rel] 改成「复用旧 sha1/chunks、取新路径当前 stat」的 stamp，
      让 commit_manifest 以新路径写回、total_chunks 不丢失。
    """
    from collections import defaultdict

    # 用 (sha1, size) 分桶：removed 侧读旧清单，added 侧读本轮算出的 stamp
    rem_by_key: dict[tuple, list[str]] = defaultdict(list)
    for rel in removed:
        rec = old.get(rel, {})
        rem_by_key[(rec.get("sha1"), rec.get("size"))].append(rel)

    add_by_key: dict[tuple, list[str]] = defaultdict(list)
    for rel in added_rels:
        st = stamps.get(rel)
        if st is not None:
            add_by_key[(st.sha1, st.size)].append(rel)

    moved: list[tuple[str, str]] = []
    consumed_added: set[str] = set()
    consumed_removed: set[str] = set()
    for key, rem_list in rem_by_key.items():
        # 排序保证确定性，一对一消费，不重复复用
        for old_rel, new_rel in zip(sorted(rem_list), sorted(add_by_key.get(key, []))):
            moved.append((old_rel, new_rel))
            consumed_added.add(new_rel)
            consumed_removed.add(old_rel)

    if not moved:
        return []

    added_rels[:] = [r for r in added_rels if r not in consumed_added]
    removed[:] = [r for r in removed if r not in consumed_removed]

    for old_rel, new_rel in moved:
        rec = old.get(old_rel, {})
        st = rel_of[new_rel].stat()
        stamps[new_rel] = FileStamp(
            size=st.st_size,
            mtime_ns=st.st_mtime_ns,
            sha1=rec.get("sha1", stamps[new_rel].sha1),
            chunks=rec.get("chunks", 0),
        )
    return moved


def compute_delta(cfg: Settings, current: tuple[Path, ...]) -> tuple[Delta, dict[str, FileStamp]]:
    """对比磁盘现状与 manifest，返回 (变更集, 本次实际算出的 stamp 映射)。

    注意：sha1 只对「mtime 或 size 变了 / 新文件」计算，未变文件直接复用
    清单里的旧 sha1，把读取量压到最低。
    """
    old = load_manifest(cfg)["files"]
    rel_of = {str(p.relative_to(cfg.docs_root)): p for p in current}

    added_rels: list[str] = []
    changed: list[Path] = []
    unchanged: list[Path] = []
    stamps: dict[str, FileStamp] = {}

    for rel, p in rel_of.items():
        rec = old.get(rel)
        st = p.stat()
        if rec is None:
            added_rels.append(rel)
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

    removed = [rel for rel in old if rel not in rel_of]
    moved = _detect_moves(added_rels, removed, old, rel_of, stamps)
    added = tuple(rel_of[r] for r in added_rels)
    return (Delta(added, tuple(changed), tuple(removed),
                  tuple(unchanged), tuple(moved)), stamps)


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
