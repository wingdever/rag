"""delta index state infer and manifest rewrite"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings


@dataclass(frozen=True)
class FileStamp:
    size: int
    mtime: int
    sha1: str
    chunks: int
    ino: int = 0


@dataclass(frozen=True)
class Delta:
    matched_pairs: tuple[Path, ...]
    matched_embeds: tuple[Path, ...]
    matched_prims: tuple[Path, ...]
    added_embeds: tuple[Path, ...]
    moved_pairs: tuple[tuple[str, str], ...]
    moved_embeds: tuple[tuple[str, Path], ...]
    moved_prims: tuple[tuple[str, str], ...]
    left_removed: tuple[str, ...]
    added_rels: tuple[str, ...] = ()
    removed_paths: tuple[str, ...] = ()

    @property
    def added(self) -> tuple[Path, ...]:
        return self.added_embeds

    @property
    def changed(self) -> tuple[Path, ...]:
        return self.matched_embeds + tuple(p for _old, p in self.moved_embeds)

    @property
    def removed(self) -> tuple[str, ...]:
        return self.left_removed

    @property
    def moved(self) -> tuple[tuple[str, str], ...]:
        return self.moved_prims

    @property
    def unchanged_view(self) -> tuple[Path, ...]:
        return self.matched_prims


def _make_stamp(path: Path, chunk_count: int = 0) -> FileStamp:
    st = path.stat()
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return FileStamp(st.st_size, st.st_mtime_ns, digest.hexdigest(),
                     chunk_count, st.st_ino)


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
        return {"version": 1, "files": {}}


def save_manifest(cfg: Settings, data: dict) -> None:
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cfg.manifest_path)


def _classify_matched(rel: str, path: Path, prev_rec: dict,
                      stamps: dict[str, FileStamp]) -> str:
    st = path.stat()
    cur_mtime = st.st_mtime_ns
    cur_size = st.st_size
    old_mtime = prev_rec.get("mtime", prev_rec.get("mtime_ns"))
    old_size = prev_rec.get("size")
    old_sha1 = prev_rec.get("sha1")
    chunks = prev_rec.get("chunks", 0)

    if cur_mtime == old_mtime:
        verdict = "prim"
        sha1 = old_sha1
    elif cur_size != old_size:
        verdict = "embed"
        sha1 = _make_stamp(path, chunks).sha1
    else:
        cur_sha1 = _make_stamp(path, chunks).sha1
        sha1 = cur_sha1
        verdict = "embed" if cur_sha1 != old_sha1 else "prim"

    stamps[rel] = FileStamp(cur_size, cur_mtime, sha1, chunks, st.st_ino)
    return verdict


def compute_delta(cfg: Settings, current: tuple[Path, ...]) -> tuple[Delta, dict[str, FileStamp]]:
    # 1. load manifest, build path maps
    prev_raw = load_manifest(cfg)["files"]
    rel_to_path = {str(p.relative_to(cfg.docs_root)): p for p in current}

    old_paths = sorted(prev_raw.keys())
    prev_by_path = prev_raw
    new_paths = sorted(rel_to_path.keys())

    # 2. path-order match -> matched / added / removed
    matched: list[Path] = []
    added_rels: list[str] = []
    removed_paths: list[str] = []
    i = j = 0
    while i < len(new_paths) and j < len(old_paths):
        n, o = new_paths[i], old_paths[j]
        if n == o:
            matched.append(rel_to_path[n])
            i += 1
            j += 1
        elif n < o:
            added_rels.append(n)
            i += 1
        else:
            removed_paths.append(o)
            j += 1
    while i < len(new_paths):
        added_rels.append(new_paths[i])
        i += 1
    while j < len(old_paths):
        removed_paths.append(old_paths[j])
        j += 1

    # stamp added files (matched stamps computed lazily in step 4/5)
    stamps: dict[str, FileStamp] = {}
    for rel in added_rels:
        stamps[rel] = _make_stamp(rel_to_path[rel])

    def _ino_of_added(rel: str) -> int:
        try:
            return rel_to_path[rel].stat().st_ino
        except OSError:
            return 0

    def _ino_of_removed(rel: str) -> int:
        return prev_by_path.get(rel, {}).get("ino", 0)

    # 3. inode match -> moved candidates (deduct from added / removed)
    moved_pairs: list[tuple[str, str]] = []
    add_sorted = sorted(added_rels, key=_ino_of_added)
    rem_sorted = sorted(removed_paths, key=_ino_of_removed)
    consumed_added: set[str] = set()
    consumed_removed: set[str] = set()
    ai = ri = 0
    while ai < len(add_sorted) and ri < len(rem_sorted):
        ia, ir = _ino_of_added(add_sorted[ai]), _ino_of_removed(rem_sorted[ri])
        if ia == 0 or ir == 0:
            if ir == 0:
                ri += 1
            else:
                ai += 1
            continue
        if ia == ir:
            moved_pairs.append((rem_sorted[ri], add_sorted[ai]))
            consumed_added.add(add_sorted[ai])
            consumed_removed.add(rem_sorted[ri])
            ai += 1
            ri += 1
        elif ia < ir:
            ai += 1
        else:
            ri += 1

    added_rels[:] = [r for r in added_rels if r not in consumed_added]
    removed_paths[:] = [r for r in removed_paths if r not in consumed_removed]

    # 4. classify matched -> matched_embeds / matched_prims
    matched_embeds: list[Path] = []
    matched_prims: list[Path] = []
    for path in matched:
        rel = str(path.relative_to(cfg.docs_root))
        verdict = _classify_matched(rel, path, prev_by_path[rel], stamps)
        if verdict == "embed":
            matched_embeds.append(path)
        else:
            matched_prims.append(path)

    # 5. classify moved candidates -> moved_embeds / moved_prims
    moved_embeds: list[tuple[str, Path]] = []
    moved_prims: list[tuple[str, str]] = []
    for old_rel, new_rel in moved_pairs:
        new_path = rel_to_path[new_rel]
        verdict = _classify_matched(new_rel, new_path, prev_by_path.get(old_rel, {}), stamps)
        if verdict == "embed":
            moved_embeds.append((old_rel, new_path))
        else:
            moved_prims.append((old_rel, new_rel))

    # 6. assemble added_embeds
    added_embeds = [rel_to_path[r] for r in added_rels]

    # 7. left_removed = removed minus moved_prims old paths
    moved_old = {old_rel for old_rel, _ in moved_prims}
    left_removed = tuple(r for r in removed_paths if r not in moved_old)

    delta = Delta(
        matched_pairs=tuple(matched),
        matched_embeds=tuple(matched_embeds),
        matched_prims=tuple(matched_prims),
        added_embeds=tuple(added_embeds),
        moved_pairs=tuple(moved_embeds) + tuple(moved_prims),
        moved_embeds=tuple(moved_embeds),
        moved_prims=tuple(moved_prims),
        left_removed=left_removed,
        added_rels=tuple(added_rels),
        removed_paths=tuple(removed_paths),
    )
    return delta, stamps


@dataclass(frozen=True)
class _EmbedOp:
    # One re-embed op: embed at new_path, delete old node keyed by old_rel (None = pure add).
    old_rel: str | None
    new_path: Path


@dataclass(frozen=True)
class ApplyPlan:
    # Index op plan for Delta (docs/dev/delta.md §6). No llama-index dependency.
    embed: tuple[_EmbedOp, ...]
    relabel: tuple[tuple[str, str], ...]
    delete: tuple[str, ...]


def plan_apply(delta: Delta, docs_root: Path) -> ApplyPlan:
    # Translate Delta into index ops (docs/dev/delta.md §6).
    embed_ops: list[_EmbedOp] = []
    # 1. added -> embed ops (no old key to delete)
    for p in delta.added_embeds:
        embed_ops.append(_EmbedOp(None, p))
    # 2. matched_embeds -> embed ops (delete same key)
    for p in delta.matched_embeds:
        embed_ops.append(_EmbedOp(str(p.relative_to(docs_root)), p))
    # 3. moved_embeds -> embed ops (delete old key)
    for old_rel, new_path in delta.moved_embeds:
        embed_ops.append(_EmbedOp(old_rel, new_path))
    # 4. assemble plan: pure renames -> relabel, left_removed -> delete
    return ApplyPlan(
        embed=tuple(embed_ops),
        relabel=tuple(delta.moved_prims),
        delete=tuple(delta.left_removed),
    )


def commit_manifest(cfg: Settings, stamps: dict[str, FileStamp],
                    chunk_counts: dict[str, int]) -> dict:
    files = {}
    for rel, st in stamps.items():
        files[rel] = {
            "size": st.size,
            "mtime": st.mtime,
            "sha1": st.sha1,
            "chunks": chunk_counts.get(rel, st.chunks),
            "ino": st.ino,
        }
    data = {"version": 1, "files": files}
    save_manifest(cfg, data)
    return data
