# ragdocs 操作手册

个人知识库（`~/Documents`）的语义检索。所有命令在项目根目录执行：
`cd ~/Projects/rag`

## 命令行召回文档（输入 query，拿回相关片段）

```bash
.venv/bin/python scripts/query.py search "你的查询词" -k 5
```

把引号里的内容换成要检索的 query 即可，例如：

```bash
.venv/bin/python scripts/query.py search "控制论里的反馈是什么意思"
.venv/bin/python scripts/query.py search "transformer 的注意力机制" -k 10
```

参数：

| 参数 | 含义 | 默认 |
|---|---|---|
| `search "query"` | 位置参数，要检索的查询文本（必填） | — |
| `-k` / `--top-k` | 返回几条召回文档 | 5 |
| `-t` / `--threshold` | 相似度阈值，低于则过滤（0 表示不过滤） | 0.0 |

输出格式（纯文本，每条一段）：

```
[1] 相对路径  (主题, score=0.5860)
------------------------------------------------------------
召回的原文片段（最多 600 字）
```

- `search` 只做向量检索，不调用 LLM、不依赖 Ollama，最快最稳。
- 结果按相似度降序排列；`score` 为余弦相似度，越大越相关。
- 若 `知识库中未找到相关内容。`，可调大 `-k` 或换更贴近原文的词。
- 阈值来自 `config/settings.yaml` 的 `query.similarity_threshold`，命令行 `-t` 可临时覆盖。

## 其他子命令

```bash
# 检索 + 生成式回答（需本地 Ollama LLM，附引用来源）
.venv/bin/python scripts/query.py ask "我的笔记里怎么理解控制论？" -k 5

# 查看索引规模 / 模型 / 主题分布
.venv/bin/python scripts/query.py stats
```

## 更新 index 向量数据库

笔记有增删改后，增量重建索引（基于 `ragdocs/loader/delta.py` 的状态推算，见 `docs/dev/delta.md`）：

```bash
.venv/bin/python scripts/build_index.py
```

首次可先 `--dry-run` 预览变更集（新增/修改/删除/移动/未变），不真正 embedding；`--only` / `--limit` 可小批量验证。
换 embedding 模型需先删 `data/` 再全量重建。
