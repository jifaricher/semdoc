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

## 保留列（自动存在，不可在 [fields] 声明）

每张表的物理 schema 由两部分组成：**保留列**（管线自身需要）+ **[fields] 用户字段**。保留列固定存在、名字不可占用：

| 列 | 类型 | 作用 | 赋值/计算方式 |
|---|---|---|---|
| `id` | Utf8 NOT NULL | 行主键（BTree 索引），merge_insert upsert 的合并键 | parent 行 = `blake3(全文)`；叶子 chunk 行 = `blake3(parent_id:chunk_index:chunk文本)`（确定性哈希，同内容重写产生同 id → 幂等 upsert） |
| `raw_text` | Utf8 NOT NULL | 该行文本内容 | parent = 原始全文；叶子 = 该 chunk 的切片文本（默认 512 字符、50 重叠，可用 `SEMDOC_CHUNK_SIZE/OVERLAP` 调整） |
| `chunk_level` | UInt8 NOT NULL | 行层级 | 0 = 父文档行；1 = 叶子 chunk 行（ANN 检索只在这一层） |
| `chunk_index` | UInt32 NOT NULL | chunk 在父文档内的 0 基序号 | 切块器按顺序分配；`expand_to=auto` 的句窗口（±1 相邻合并）依赖它 |
| `parent_doc_id` | Utf8 NULL | 叶子行指向的父文档 id | 叶子行 = 其 parent 的 id；父行 = NULL。删除/展开/级联更新都靠它 |

## source_path / source_type：现在是普通字段

旧版本（含 semrag）里它们是保留列；**现在降级为普通 `[fields]` 字段**——需要来源追溯的库自己声明，不需要的库不存在这两列：

```toml
[fields]
# 来源路径：建议 index = true 便于过滤；replace_key = true 启用"同来源替换"
source_path = { type = "string", index = true, replace_key = true }
# 来源形态：file / text / directory / glob ...
source_type = { type = "string", replace_key = true }
```

三个写入入口（CLI `add`、server `POST /documents`、MCP `add_document`）都会在 schema 声明了这两个字段时自动填值（`source_path` 取 `--source`/请求参数，`source_type` 分别为 `file`/`text`/请求参数），MCP/REST 的 metadata 里也可显式覆盖。

## replace_key：按业务身份替换（replace-on-add）

id 是内容哈希——改了正文再 add 会得到新 id，旧版本不会被触碰。要"修改文档"语义，在 schema 里给业务标识字段加 `replace_key = true`（可多个，**AND 语义**：所有键都匹配才替换）：

- `add` 写入前，用本次写入的 replace_key 值组合成等值查询，命中同一批旧 parent 文档（及其全部叶子 chunk）→ 先删除（LanceDB 级联 + lightrag 镜像同步删除），再写新版本
- 所有 replace_key 字段都必须是 `string` 类型（validate 强制）
- 未传齐 replace_key 字段的写入不做替换（追加语义）
- MCP `add_document` 响应会注明 `（已按 replace_key 替换旧版本 <旧id>）`

## schema 与数据库的兼容性校验

`semdoc init` 时会把物理 `[fields]`（名 -> 类型）快照写到 `<db>/physical_fields.toml`。之后每次打开库（CLI / server / MCP），传入的 schema 都会和快照比对：

- 列集与类型**完全一致** → 正常打开（`index`/`required`/`replace_key` 等元数据可以随 schema 演进）
- schema 删了库里的列、加了库里没有的列、或改了列类型 → **启动即报错**，逐项列出差异（提示 re-init 或 semdoc-migrate）
- 旧库没有快照文件 → 打印一次警告并跳过检查（向后兼容），重新 init 后启用

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
