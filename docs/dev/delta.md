# delta — 增量索引的状态推算与清单读写

`ragdocs/loader/delta.py` 的设计与实现文档。**代码是事实来源**，本文档通过具体符号
反向索引代码，不在代码里反向索引本文档。

## 代码索引

| 符号 | 所在文件 | 说明 |
|------|----------|------|
| `FileStamp` | `ragdocs/loader/delta.py` | 单文件指纹，落 manifest |
| `Delta` | `ragdocs/loader/delta.py` | 一次增量计算的结果（§1） |
| `compute_delta` | `ragdocs/loader/delta.py` | 主流程，对应 §3 步骤 1–7 |
| `_classify_matched` | `ragdocs/loader/delta.py` | 四分支 content 判定，对应 §3 步骤 4 |
| `_make_stamp` / `load_manifest` / `save_manifest` / `commit_manifest` | `ragdocs/loader/delta.py` | 指纹计算与清单读写 |
| `ApplyPlan` / `_EmbedOp` | `ragdocs/loader/delta.py` | `Delta` 的索引操作计划（§6） |
| `plan_apply` | `ragdocs/loader/delta.py` | 把 `Delta` 翻译成 `ApplyPlan`（§6） |
| `RagIndex.build` | `ragdocs/core/index.py` | 执行 `ApplyPlan`：embed / 删 / 重贴（§3 步骤 6–7） |
| `_relabel_moved` | `ragdocs/core/index.py` | 对 `moved_prims` 就地重贴 metadata |

> 旧实现 `ragdocs/loader/manifest.py`（含 `_detect_moves`、旧 `Delta`）已弃用，不在此讨论。

---

# 当前设计

## 1. 数据结构

### `FileStamp`（ragdocs/loader/delta.py）
```
size: int
mtime: int        # os.stat().st_mtime_ns；落盘键名 mtime，旧键 mtime_ns 读时兼容
sha1: str
chunks: int
ino: int = 0     # 仅作 mv 候选配对的易失 hint + 审计，非主键；旧记录缺字段按 0
```

### `Delta`（ragdocs/loader/delta.py）
结构关系：
```
{new_paths − old_paths}(path) = matched_pairs + (add_paths, del_paths)
  matched_pairs = matched_embeds + matched_prims
{add_paths − del_paths}(ino)  = moved_pairs + (added_embeds, left_removed)
  moved_pairs  = moved_embeds  + moved_prims
```
更新处理（`RagIndex.build` 据此执行）：
```
matched_embeds / moved_embeds / added_embeds : embed + 重算 sha1 + 刷 metadata
matched_prims  / moved_prims                : 仅刷 metadata（不重 embed）
left_removed                               : delete
```
字段：
```
matched_pairs : tuple[Path, ...]            # 同 path 还在 = matched_embeds + matched_prims
matched_embeds: tuple[Path, ...]            # 同路径、内容变，同路径重 embed
matched_prims : tuple[Path, ...]            # 同路径、内容未变（仅刷 metadata）
added_embeds  : tuple[Path, ...]            # 新增文件，直接 embed 插入
moved_pairs   : tuple[(str, str), ...]      # = moved_embeds + moved_prims（派生并）
moved_embeds  : tuple[(str, Path), ...]     # (旧路径, 新路径)：改名且内容变，删旧+重 embed 新路径
moved_prims   : tuple[(str, str), ...]      # (旧路径, 新路径)：纯改名、内容未变，仅重贴 metadata
left_removed  : tuple[str, ...]             # 消失路径（相对，已扣 moved 旧路径）
added_rels    : tuple[str, ...]             # 内部用：新增相对路径视图
removed_paths : tuple[str, ...]             # 内部用：消失相对路径
```
`Delta` 另含兼容视图属性 `added` / `changed` / `removed` / `moved`(=moved_prims) / `unchanged_view`(=matched_prims)。

## 2. 字段语义（本工作流：本地 vscode 编辑 + git push/pull，单 fs）

| 字段  | 变了说明 | 没变说明 |
|-------|----------|----------|
| mtime | 编辑器写 | **本工作流前提下视为内容未变** |
| size  | 内容必变 | 等长改写 |
| sha1  | 内容必变 | 内容未变 |
| inode | **不参与 content 判定** 本环境 edit/mv 均不变 |

- `mtime`/`size` 是廉价探针：决定是否值得算 sha1。
- `sha1` 是唯一权威：只有相同才能确认内容未变。

## 3. compute_delta 主流程（ragdocs/loader/delta.py::compute_delta，对应步骤 1–7）

1. **取两表**：旧 manifest 的 `files` 键（old_paths）与当前扫描 `current`（new_paths），各按 path 排序；
   `prev_by_path` 以 path 为主键保存上一轮记录。
2. **同序匹配**（path）：双指针逐对比较，相同→`matched_pairs`，仅新侧→`added_rels`，仅旧侧→`removed_paths`。
3. **取移动**（inode）：复用步骤 2 骨架，比较键换为 inode。两侧按 inode 排序后双指针比对，
   inode 相同即配对为 `moved_pairs` 候选（不做 sha1/size 校验、无第二趟回退）；配对上的从
   `added_rels`/`removed_paths` 扣减。内容是否真变留到步骤 5 判定。
4. **matched_pairs 判定变更**（`_classify_matched`，见 §4）：
   ```
   mtime 相同                          -> prim    (不读文件算 sha1)
   mtime 不同 + size 不同             -> embed
   mtime 不同 + size 同 + sha1 不同   -> embed   (sha1 现算)
   其余 (mtime 不同 + size/sha1 同)    -> prim
   ```
   - embed → `matched_embeds`（同路径重 embed）
   - prim → `matched_prims`（仅刷 metadata；含 mtime 同/异两种情形）
5. **moved_pairs 内容再判定**（同样走 `_classify_matched`）：inode 同序只是候选配对，改名同时编辑时
   inode 不变但内容可能变，必须识别：
   - embed → `moved_embeds`（(旧路径, 新路径)：按旧路径删节点 + 重 embed 新路径）
   - prim → `moved_prims`（(旧路径, 新路径)：纯改名，仅重贴 metadata）
   拆分后 `moved_pairs = moved_embeds + moved_prims`（派生并）。
6. **embed 三类列表**（`RagIndex.build`）：`added_embeds` 直接插入；`matched_embeds` 先按该 path 删节点再插入；
   `moved_embeds` 按旧路径删、按新路径插入。三类内容均变更，写回 manifest 时重算 sha1 并刷新
   size/mtime（added 走 `_make_stamp` 全量算；matched/moved 由 `_classify_matched` 在 embed 时写入）。
   `matched_prims`/`moved_prims` 仅刷 metadata。
7. **delete** `left_removed`（`removed_paths` 扣去 `moved_prims` 旧路径）。

## 4. _classify_matched 四分支判定（ragdocs/loader/delta.py::_classify_matched）

返回 `"embed"` / `"prim"` 两态之一，并把当前 `size/mtime/sha1/ino` 写入 `stamps[rel]`。
分支见 §3 步骤 4 的四行表（mtime 相同直接 prim，连 sha1 都不算；这是相对旧实现的关键区别，
代价是接受 git checkout/pull 回同版本等极低概率漏判，见 §2 前提）。

## 5. 已知边界

**忽略（事件推演出错，但向量库结果正确）**
1. 多文件内容整体交换：sha1 复核会触发双 changed，向量正确。

**异常（接受，可能导致向量滞后）**
- mtime 相同但内容变（git checkout/pull 回同版本等）：本工作流概率极低，已显式接受；未重 embed，
  直至下次 size/mtime 变动才修正。
- mtime 不同但内容不变（vscode 空保存、git pull 仅动时间）：由 §3 步骤 4 末支正确识别为 prim。

## 6. 消费层（ragdocs/loader/delta.py::plan_apply / ApplyPlan）

`compute_delta` 只推算出 `Delta`，本身不触碰向量库。`plan_apply(delta, docs_root)` 把 `Delta`
翻译成**无 llama-index 依赖**的索引操作计划 `ApplyPlan`，由 `ragdocs/core/index.py::RagIndex.build`
执行（embed / 删 / 重贴）。这样「变更事件如何映射到索引动作」的单一事实来源仍在 `delta.py`。

`ApplyPlan` 字段：
```
embed   : tuple[_EmbedOp, ...]   # 需重 embed 的文件
relabel : tuple[(str, str), ...]  # 纯改名对（moved_prims），仅重贴 metadata
delete  : tuple[str, ...]         # 需直接删除的旧路径（left_removed）
```
`_EmbedOp`：
```
old_rel : str | None   # 重 embed 前需先删除的旧节点键；None = 纯新增，无旧节点
new_path: Path         # 重新 embed 并插入的新路径
```

映射规则（对应 §3 步骤 6–7）：
- `added_embeds`     -> `_EmbedOp(None, p)`
- `matched_embeds`   -> `_EmbedOp(rel, p)`（rel == 新路径，删同键旧节点后重插）
- `moved_embeds`     -> `_EmbedOp(old_rel, new_path)`（按旧路径删、按新路径重 embed）
- `moved_prims`      -> `relabel`
- `left_removed`     -> `delete`

`RagIndex.build` 据此：`plan.embed` 解析+批量 embed 后 insert；`stale_ids` = 各 `embed.op.old_rel`
（非 None）+ `plan.delete` 先删；`_relabel_moved(plan.relabel)` 仅重贴 metadata；最后 `commit_manifest`
写回 `stamps`。

