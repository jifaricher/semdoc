# semdoc — 配置驱动 Schema 的向量知识库

semdoc 是 semrag 的第二版：一个通用向量知识库，**schema 是配置文件而不是硬编码结构体**。一份 schema 文件对应一个数据库；存储、检索、嵌入、重排、图谱推理全部可插拔。

## 核心概念

| 配置文件 | 职责 | 是否随库迁移 |
|---|---|---|
| `schema.toml` | 库 schema：表名、向量列、用户标量字段、插件声明 | **是**（存在 `<db>/schema.toml`） |
| `Config.toml` | 部署事实：embedder/reranker 端点、chunk 大小、监听地址、TLS | 否（跟着环境走） |
| `lightrag.toml` | lightrag-server 进程配置：存储后端、LLM/Embedding/Rerank 绑定 | 否（独立解耦） |

优先级：CLI flag > 环境变量（`SEMDOC_*`）> `Config.toml` > 内置默认。密钥永不落盘：配置里只写 `*_env`（环境变量名），值本身放环境变量。

## 三种接口

### 1. CLI（`semdoc`）

```bash
semdoc init  --schema schema.toml --db /path/to/db    # 初始化库
semdoc add   --db /path/to/db --file doc.md [--meta K=V]
semdoc query --db /path/to/db -t "查询" [--mode semantic|fts|rerank] [--filter '{"domain":"mm"}']
semdoc stats --db /path/to/db
```

### 2. HTTP REST（`semdoc-server`）

```
POST /documents             {text, source_path?, meta?}   写入（自动嵌入）
POST /documents/delete      {id}                          删除（级联图谱）
GET  /documents/:id         精查一条（raw_text 不截断）
POST /query/semantic        {text, limit, filter?, expand_to?}
POST /query/text            {text, limit, filter?}        BM25 全文检索
POST /query/reranked        {text, limit, filter?}        cross-encoder 精排
POST /query/graph           {text, limit, answer?}        图谱检索（不可用自动降级）
POST /query/hybrid          {text, limit, answer?}        语义+图谱并行合并
GET  /stats                 统计
GET  /health                健康检查
```

REST 鉴权：`--auth` + `server.token_env`（或 `SEMDOC_TOKEN`）→ 所有接口要求 `Authorization: Bearer <token>`。

### 3. MCP

**两种传输，同一套工具面**（业务逻辑在 `src/mcp.rs`，单一事实源）：

| 二进制 | 传输 | 鉴权 | 会话 |
|---|---|---|---|
| `semdoc-mcp` | stdio（JSON-RPC over stdin/stdout） | 无需（本地管道无网络暴露） | 无 |
| `semdoc-server /mcp` | **streamable HTTP**（2025-03-26 spec） | **必须** `SEMDOC_MCP_TOKEN`（不设则 /mcp 全部 401，fail-closed） | `initialize` 签发 `Mcp-Session-Id`；**24h 空闲超时**（活跃不过期）；SQLite 持久化（`<db>/mcp_sessions.sqlite3`），重启不掉会话；`DELETE /mcp` 主动终止 |

HTTP MCP 生命周期：

```bash
export SEMDOC_MCP_TOKEN=<secret>
# 1) initialize → 响应头拿 Mcp-Session-Id
curl -X POST http://host:8092/mcp -H 'Authorization: Bearer <secret>' \
     -H 'Accept: application/json' \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}'
# 2) 后续请求带 Mcp-Session-Id 头；tools/list 看工具；tools/call 调用
# 3) DELETE /mcp + Mcp-Session-Id 主动结束会话
```

## MCP 工具清单（11 个）

| 工具 | 说明 |
|---|---|
| `query_semantic` | 语义检索（small-to-big：叶子 chunk ANN，按 `expand_to` 展开 chunk/parent/auto） |
| `query_text` | BM25 全文检索，适合精确术语/函数名 |
| `query_reranked` | ANN 召回 80 条 + cross-encoder 精排（需 `--rerank`，故障自动降级） |
| `query_graph` | 图谱检索。默认返回结构化实体/关系/chunk+映射本地文档；`answer=true` 走 LLM 综合答案（慢） |
| `query_hybrid` | 语义（reranked）与图谱并行，按 id 合并去重 |
| `get_document` | 按 id 精查，raw_text 完整不截断 |
| `add_document` | 写入：切块+自动嵌入+入库；同文本幂等 upsert（修改正文后刷新向量的正确方式）；镜像到图谱 |
| `delete_document` | 删除：LanceDB 父文档+全部叶子；级联删除 lightrag 文档（内部 id 自动推导） |
| `update_document_metadata` | **按 schema 原位修补元数据字段**：只接受 `[fields]` 声明的标量字段并校验类型；不碰 raw_text/向量列（零重嵌入）；`cascade=true`（默认）连子 chunk 一起改；值为 null 清空字段 |
| `list_documents` | 分页浏览父文档（200 字符预览） |
| `stats` | 总行数 |

字段名是每个库自己的事（schema.toml `[fields]` 决定），MCP 工具描述动态列出本库可用字段——不需要硬编码 subsystem/domain 之类的固定名。

**修改正文 vs 修改标签**：正文变了 → 用相同文本重新 `add_document`（同 id 幂等，自动重嵌入）；只改标签 → `update_document_metadata`。向量列不暴露改值接口，保证向量和文本永远一致。

## 后端选型

| 能力 | 后端 |
|---|---|
| Embedding | `onnx`（bge-m3 进程内，feature `onnx` 默认开）、`http`（OpenAI 兼容 `/v1/embeddings`，如 ollama） |
| Reranker | `tei`（TEI 兼容 `/rerank`，当前环境用 FlagEmbedding 包装的 bge-reranker-v2-m3）、`openai`（`/v1/rerank`）、`onnx`（进程内 INT8）、`none` |
| Graph | `lightrag-server`（HTTP 零 Python，当前方案）、`lightrag-embedded`（PyO3，feature）、`none` |

瞬时失败重试 + 降级：reranker/lightrag 挂掉不会让查询失败，自动退回 ANN 顺序/纯语义检索。

## lightrag 交互协议（1.5.6 实测）

- 插入：`POST /documents/text` `{"text", "file_source": "<docid>.txt"}`（异步抽取实体；不接受 ids）
- 删除：`DELETE /documents/delete_document` `{"doc_ids": ["doc-md5(<docid>.txt)", "<docid>"]}`（内部 id = md5(file_source) 自动推导；pipeline busy 返回 200+status=busy，需重试）
- 查询：`POST /query`（LLM 答案）/ `POST /query/data`（结构化）；chunks[].file_path 去掉 .txt 即本库 parent_doc_id，用于映射回 LanceDB

图数据本体在 lightrag 的存储后端（本机 Postgres:5455 + Neo4j:7687，workspace `semdoc_kernel`），semdoc 只通过 HTTP 操作。

## 启动 lightrag

```bash
# 密钥注入环境（当前可从 /workspace/semrag/.env 提取）
export LIGHTRAG_PG_PASSWORD=... LIGHTRAG_NEO4J_PASSWORD=... \
       LIGHTRAG_LLM_API_KEY=... LIGHTRAG_EMBEDDING_API_KEY=... LIGHTRAG_RERANK_API_KEY=...
./scripts/start-lightrag-server.sh --bg     # 后台启动（读 lightrag.toml）
./scripts/start-lightrag-server.sh --health # 探活
./scripts/start-lightrag-server.sh --stop   # 停止
```

## 编译

```bash
cargo build --release --bins   # 二进制编译一律用这一条
cargo test --lib               # 单元+集成测试
cargo clippy --all-targets     # lint
```

## 安全基线

- MCP/HTTP 的 `limit` 上限 200；`raw_text` 响应截断 2000 字符（`get_document` 除外）
- filter DSL 类型按 schema 校验，字符串转义防注入；`filter_sql` 仅限受信输入（不暴露给 MCP）
- onnx 不可用、模型 dim 不匹配、密钥环境变量未设置 → 启动期明确报错（fail fast）
