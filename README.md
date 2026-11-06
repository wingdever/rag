# ragdocs

把 `~/Documents` 的个人知识库做成 RAG 服务，并通过 MCP 暴露给 agent 调用。

- 框架：llamaIndex 0.14（`SimpleVectorStore` 本地持久化）
- embedding：默认 **远程 embedding 服务**（OpenAI 兼容 /embeddings 协议，如讯飞 MaaS
  `Qwen3-Embedding-8B`，768 维）；llama.cpp（GGUF）/ Ollama / sentence-transformers 为
  可选后端（`config/settings.yaml` 切换）。本机无可用 GPU 时选远程最省事。
- 生成（可选）：Ollama `qwen2.5:7b`，仅 `ask_docs` 用；不下 LLM 也能只用 `search_docs` 检索
- 协议：MCP 2.3，stdio 传输
- 规模：2440 个纯文本文件 → ~2881 个 chunk

## 快速开始

```bash
cd ~/Projects/rag

# 1. embedding 后端：默认 remote（无需本机 GPU）。
#    密钥不放 config/settings.yaml（避免提交泄露），二选一：
#      a) 建 user/secrets.yaml（已被 .gitignore 忽略），内容：
#           remote_embed:
#             api_key: "你的讯飞key"
#      b) 或 export XF_YUN_API_KEY="你的key"
#    优先级：user/secrets.yaml > 环境变量 XF_YUN_API_KEY > settings.yaml 占位。
#    base_url / model / batch_size 等在 config/settings.yaml 里配。
#    （若改回 llama.cpp：需先装 llama-cpp-python 并准备 GGUF 模型，见 docs/06）

# 2. 建索引（首次全量，之后增量；--only/--limit 可先小批量验证）
.venv/bin/python scripts/build_index.py --dry-run   # 先预览
.venv/bin/python scripts/build_index.py

# 3. 命令行验证（search_docs 不需要 LLM，也不依赖 Ollama）
.venv/bin/python scripts/query.py search "控制论" -k 5
.venv/bin/python scripts/query.py ask "我的笔记里怎么理解控制论？"   # 需 Ollama LLM

# 4. 挂到 agent
#    把 config/mcp.example.json 里的 ragdocs 一项合并进 ~/.workbuddy/mcp.json
#    的 mcpServers（不要覆盖已有的 gitee 配置），重启 agent
```

## 暴露给 agent 的工具

| 工具 | 作用 |
|---|---|
| `search_docs` | 语义检索，返回原文片段 + 文件路径 + 分数（不调 LLM） |
| `ask_docs` | 检索 + 本地 LLM 生成答案，附引用来源 |
| `reindex_docs` | 增量重建索引（改完笔记后让 agent 自己刷新） |
| `rag_stats` | 索引规模 / 模型 / 主题分布 / 隐私开关状态 |

## 配置

全部在 `config/settings.yaml`：模型名、分块参数、索引扩展名白名单、
检索 top_k / 阈值、系统提示词、隐私排除开关。改完不用动代码。

> **隐私**：按你的决定当前是**全部索引**（`privacy.exclude_enabled: false`），
> 含 `4.0-selfInfo`、`passwords.csv`、简历等。风险与一键收紧方法见
> [docs/05-隐私与安全.md](docs/05-隐私与安全.md)。

## 学习资料

从 [docs/00-总览与学习路径.md](docs/00-总览与学习路径.md) 开始，共 7 篇，
讲清每个设计决策的「为什么」，以及本次开发中实际踩到的坑
（Python 3.14 wheel、pypi 镜像、mcp 2.x 改名、`ref_doc_id` 删除语义、stdout 协议通道等）。

## 运行与排障

见 [docs/06-运行与排障.md](docs/06-运行与排障.md)。

## 与库内已有 `docindex` 的关系

`~/Documents/0.0-how-docs/tools/docindex/` 已有一个基于 SQLite FTS5 的
**词法检索 + MCP** 工具。本项目做的是**语义检索**，两者互补、可同时挂在
agent 上（server 名不同）。详见 [docs/00 第五节](docs/00-总览与学习路径.md)。

## 已知限制

- docx / pdf / xlsx / 老式 .doc 未索引（第一阶段纯文本优先，清单在 `data/skipped_files.txt`）
- 换 embedding 模型需手动删 `data/` 全量重建
- 改名/移动文件不重 embed（先按 inode 号 `ino` 本机配对、失配回退内容 sha1，见 docs/03），
  只更新路径元数据：检索结果路径是新值，但向量内嵌的旧路径/标题字段不刷新，
  `doc_id` 仍为旧路径哈希；要完全一致需触发内容变更重算或删 `data/` 全量重建
- manifest / 节点里的 `ino`（inode 号，`st_ino`）是**本机易失标识**，仅用于本机改名快速配对与
  审计，不做跨机/同步身份；跨文件系统、git checkout、rclone 同步后会失效并自动回退 sha1。
  本机 `~/Documents` 全在同一分区，`st_dev` 恒为常量、属冗余，故只存 `ino` 不存 dev
- `ask_docs` 为单次检索+生成，无多轮对话
- 无并发写保护；单人使用足够

## License

GNU General Public License v3.0 — see `LICENSE`.
