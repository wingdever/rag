# data-struct

增量索引的状态推算与清单读写（即 `ragdocs/loader/delta.py`）的设计与实现文档，已移至
[delta.md](./delta.md)。

- 数据结构 `FileStamp` / `Delta`、主流程 `compute_delta`、判定 `_classify_matched`、消费方
  `RagIndex.build` / `_relabel_moved`：见 [delta.md](./delta.md)。
- 旧实现 `ragdocs/loader/manifest.py` 已弃用，不在此讨论。
