"""文件发现：遍历文档库，产出「待索引文件」清单与「跳过清单」。

职责边界（刻意划清）：
- 本模块只决定「哪些文件进索引」，不读内容、不切分。
- 过滤规则全部来自 Settings，方便审计：
  扩展名白名单 / 目录黑名单 / 单文件大小上限 / 隐私排除开关。

为什么用白名单而不是黑名单：个人知识库里混着 .git、二进制附件、
17MB 的 docs.db。白名单让「新增一种奇怪格式」默认不进索引，
比事后补救安全。
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings


@dataclass(frozen=True)
class SkipReason:
    path: Path
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    files: tuple[Path, ...]
    skipped: tuple[SkipReason, ...]


def _matches_privacy(rel: str, patterns: tuple[str, ...]) -> bool:
    """隐私规则匹配，支持两种写法：

    - 裸串（无通配符）：作为相对路径的子串，如 "4.0-selfInfo"、"passwords.csv"
    - glob 串：把 "**" 展开成 "*" 后用 fnmatch 匹配整条相对路径，
      再用同样的模式匹配文件名（覆盖 "**/简历*" 只写文件名的情形）。

    fnmatch 的 "*" 本身就能跨 "/"，所以 "**" 退化成 "*" 不影响正确性，
    只是少了一层「必须跨目录」的语义——对排除规则来说宁可多排，不漏排。
    """
    name = Path(rel).name
    for pat in patterns:
        if not pat:
            continue
        if not any(ch in pat for ch in "*?["):
            if pat in rel:
                return True
            continue
        expanded = pat.replace("**", "*")
        if fnmatch.fnmatch(rel, expanded) or fnmatch.fnmatch(name, expanded):
            return True
    return False


def discover(cfg: Settings) -> DiscoveryResult:
    """同步遍历。5704 个文件量级下 os.walk 足够，无需异步。"""
    root = cfg.docs_root
    if not root.is_dir():
        raise FileNotFoundError(f"文档库根目录不存在: {root}")

    found: list[Path] = []
    skipped: list[SkipReason] = []

    for dirpath, dirnames, filenames in root.walk():
        # 目录黑名单：原地修改 dirnames 可让 walk 不再深入，效率高
        dirnames[:] = [d for d in dirnames if d not in cfg.ingest.exclude_dirs]

        for name in filenames:
            p = dirpath / name
            rel = str(p.relative_to(root))

            if p.suffix.lower() not in cfg.ingest.include_ext:
                skipped.append(SkipReason(p, f"扩展名不在白名单: {p.suffix or '(无)'}"))
                continue

            try:
                size = p.stat().st_size
            except OSError as e:
                skipped.append(SkipReason(p, f"stat 失败: {e}"))
                continue

            if size > cfg.ingest.max_file_bytes:
                skipped.append(SkipReason(p, f"超过大小上限 {size}B > {cfg.ingest.max_file_bytes}B"))
                continue

            if size == 0:
                skipped.append(SkipReason(p, "空文件"))
                continue

            if cfg.privacy.exclude_enabled and _matches_privacy(rel, cfg.privacy.exclude_path_patterns):
                skipped.append(SkipReason(p, "隐私规则排除"))
                continue

            found.append(p)

    found.sort()
    return DiscoveryResult(files=tuple(found), skipped=tuple(skipped))
