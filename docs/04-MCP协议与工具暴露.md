# 04 MCP 协议与工具暴露：让 agent 真的能用上

对应代码：`ragdocs/mcp_server.py`、`config/mcp.example.json`

---

## 1. MCP 解决的是什么问题

没有 MCP 时，每个 agent（CodeBuddy、Claude Desktop、Cline、自研 agent）
各有各的「工具插件」写法，同一个知识库要写好几套适配。
MCP 定了一个统一协议：**agent 是 client，你的服务是 server，
双方用 JSON-RPC 说话**。

```
agent (client)                      ragdocs (server)
   |--- initialize -------------------->|
   |<-- 能力声明 -----------------------|
   |--- tools/list -------------------->|   「你有哪些工具？」
   |<-- [search_docs, ask_docs, ...] ---|
   |--- tools/call(search_docs) ------->|   「查一下 X」
   |<-- [{path, text, score}, ...] -----|
```

agent 读 `tools/list` 的返回，就知道该在什么时机、用什么参数调哪个工具。
**你的 RAG 不需要知道对面是哪个 agent。**

---

## 2. 为什么选 stdio 传输

MCP 支持 stdio / SSE / streamable-http。本项目用 **stdio**：

| | stdio | HTTP/SSE |
|---|---|---|
| 端口 | 不需要 | 要开、要管冲突 |
| 鉴权 | 不需要（进程边界即信任边界） | 要考虑 |
| 生命周期 | agent 拉起子进程，退出即停 | 要常驻、要守护 |
| 部署 | 一行 command 配置 | 要起服务 |

对个人本地知识库，stdio 明显更合适：不常驻、不占端口、不暴露到网络。

### ⚠️ 头号坑：stdout 是协议通道

agent 通过**子进程的 stdin/stdout** 收发 JSON-RPC。你在 server 里
`print("debug")` 到 stdout，那行文本会混进 JSON-RPC 流，client 解析直接失败，
表现为「连不上 / 握手超时」，且很难联想到是某行 print 引起的。

所以 `mcp_server.py` 里所有日志都走 `_log()`：

```python
def _log(msg: str) -> None:
    print(f"[ragdocs] {msg}", file=sys.stderr, flush=True)
```

stderr 不参与协议，agent 一般会把它记到自己的日志里。
**写 stdio MCP server，第一条规则就是：stdout 一个字节都别自己写。**

---

## 3. mcp 2.x 的写法（本机实测 2.3.0）

`requirements.txt` 写 `mcp>=1.0.0`，实际装到 **2.3.0**。2.x 把 1.x 的
`FastMCP` 改名为 `MCPServer`（从 `mcp.server.mcpserver` 导入）。
直接 `from mcp.server.fastmcp import FastMCP` 会得到一个明确报错，
告诉你去看迁移指南——这个报错本身挺友好。

2.x 推荐写法：**装饰器 + 普通函数**，schema 全自动：

```python
from mcp.server.mcpserver import MCPServer

server = MCPServer("ragdocs", instructions="个人知识库检索工具集")

@server.tool()
def search_docs(query: str, top_k: int = 5, threshold: float = 0.0) -> dict:
    """在个人知识库中做语义检索，返回最相关原文片段及文件路径。

    适合查找概念定义、笔记原文。只检索、不调用 LLM。

    Args:
        query: 检索问题或关键词
        top_k: 返回片段数
        threshold: 最低相似度，0 表示不过滤
    """
    ...

server.run(transport="stdio")
```

SDK 从三处自动生成工具 schema：
- **工具名 / 描述** ← 函数名 + docstring 第一段
- **参数说明** ← docstring 的 `Args:` 段
- **类型 / 默认值 / 必填** ← 类型注解 + 函数签名

实测生成的四个工具：

```
search_docs | params: query, top_k, threshold
ask_docs    | params: question, top_k
reindex_docs| params: (无)
rag_stats   | params: (无)
```

对比 1.x 要手写 `list_tools()` 返回 `Tool(inputSchema={...})` 再写
`call_tool()` 里的大 `if/elif` 分派，2.x 少了一半样板代码，
而且**签名和实现永远不会不同步**（改参数忘了改 schema 是 1.x 常见 bug）。

> 我们没有为了「和网上教程一致」去 pin `mcp<2`。实测真实 API 后按 2.x 写，
> 依赖保持当前稳定版。这个判断过程本身比代码更值得记。

---

## 4. 四个工具的设计意图

| 工具 | 为什么需要 |
|---|---|
| `search_docs` | 最常用。agent 往往只需要原文片段自己组织答案，不调 LLM 省时省 token |
| `ask_docs` | 需要跨多篇笔记综合时用；返回 `sources` 便于溯源，不黑箱 |
| `reindex_docs` | 让 agent 在「刚改完笔记」后自己刷新索引，不用你手动跑脚本 |
| `rag_stats` | 排障入口：索引在不在、多少个 chunk、用的哪个模型、隐私开关开没开 |

一个细节：工具函数里**没有 try/except 兜底**了（1.x 版我加过）。
2.x 的 `MCPServer` 会自己把未捕获异常转成 protocol 层的错误返回，
agent 能看到具体异常类型与消息。手写兜底反而会把 stack trace 吞掉、
变成一句笼统的「执行失败」，不利于排障。

另一个细节：返回 `dict` 而不是拼好的 JSON 字符串。2.x 支持
`structured_output`，SDK 会同时给出人类可读文本和结构化内容，
agent 解析更可靠。

---

## 5. 挂到 CodeBuddy：配置片段

本机 agent 配置在 `~/.workbuddy/mcp.json`，结构是：

```json
{ "mcpServers": { "gitee": { "url": "...", "headers": {...} } } }
```

**注意：不要整个文件覆盖**——你已经有 gitee 的 HTTP server。
只把 `config/mcp.example.json` 里的 `ragdocs` 一项合并进 `mcpServers`：

```json
{
  "mcpServers": {
    "gitee": { "...": "保持原样" },
    "ragdocs": {
      "command": "/home/wing/Projects/rag/.venv/bin/python",
      "args": ["-m", "ragdocs.mcp_server"],
      "cwd": "/home/wing/Projects/rag",
      "env": { "PYTHONUNBUFFERED": "1" }
    }
  }
}
```

几个字段的理由：
- `command` 用 **venv 里的绝对路径**，不能写 `python`（agent 的 PATH 未必有 venv）；
- `cwd` 必须是项目根，否则 `import ragdocs` 和配置里的相对路径解析会失败；
- `PYTHONUNBUFFERED=1`：保证 stderr 日志实时可见，也避免 stdio 缓冲导致
  握手消息迟迟不发。

> 按 Commons 共享规则，我没有直接改 `~/.workbuddy/mcp.json`。
> 这类共享配置由你本人或 agent-sync 流程处理更稳妥。

---

## 6. 不挂 agent 也能验证：用 client 直接握手

开发时不必每次重启 CodeBuddy。写个 20 行的 client 起子进程连 stdio：

```python
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

params = StdioServerParameters(
    command="/home/wing/Projects/rag/.venv/bin/python",
    args=["-m", "ragdocs.mcp_server"],
    cwd="/home/wing/Projects/rag",
)
async with stdio_client(params) as (r, w):
    async with ClientSession(r, w) as s:
        await s.initialize()
        print(await s.list_tools())
        print(await s.call_tool("rag_stats", {}))
```

本次实测输出（在 Ollama 模型还没拉、索引还没建的情况下）：

```
[OK] MCP 握手成功，工具列表: search_docs / ask_docs / reindex_docs / rag_stats
[OK] rag_stats 返回: {files_indexed: 0, index_exists: false, embed_model: "bge-m3", ...}
```

**这证明协议层完全独立**：即使底层索引是空的、模型没就绪，
MCP 这一层也能单独验证。分层 again。

---

## 7. 工具暴露之后，agent 侧会发生什么

你在 agent 里问「我的笔记里怎么看控制论？」，agent 会：
1. 读 `tools/list`，看到 `search_docs` 的描述；
2. 判断这个问题该查知识库，生成 `{"query": "控制论 理解"}` 调用；
3. 拿到 `hits[]`（含 `path` 和 `text`），把这些片段放进自己的上下文；
4. 用它自己的模型组织答案，并附上文件路径。

**关键认知**：`search_docs` 不生成答案，生成答案的是 agent 自己。
`ask_docs` 才用本地 qwen 生成。两条路径各有用途——
前者让 agent 保留全部主动权，后者适合「让知识库自己先消化一遍」。

---

## 小结
- MCP = agent 与工具的统一 JSON-RPC 协议；stdio 最适合本地单人场景；
- **stdout 属于协议，日志只走 stderr**；
- 2.x 的 `@server.tool()` 从类型注解 + docstring 自动生成 schema，别手写；
- 配置用 venv 绝对路径 + 项目根 cwd，合并而非覆盖现有 mcp.json；
- 用 `mcp.client.stdio` 可以在不重启 agent 的情况下完整验证协议层。
