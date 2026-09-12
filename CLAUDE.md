# CLAUDE.md

## 编译规则（必须遵守）

- 所有二进制编译一律使用 `cargo build --release --bins`
- 禁止 `cargo build --release` 和 `cargo build --bins`（会让磁盘 iowait 打满）
- 测试用 `cargo test --lib`；lint 用 `cargo clippy --all-targets`

## TLS（内网自签证书）

- 所有 HTTP 客户端统一走 `semdoc::tls`（src/tls.rs）；**默认从 Config.toml [tls] 读**
  （`TlsConfig::apply_env` 在配置加载时导出 env），env `SEMDOC_CA_BUNDLE` /
  `SEMDOC_TLS_INSECURE` 优先。
- 优先 `ca_bundle`（reqwest `tls_certs_merge`，公司 CA 加信任链）而非 `insecure`。

## 与 lightrag 交互（backend = "lightrag-server"）

- 启动/停止/探活一律用本项目脚本：`./scripts/start-lightrag-server.sh [--bg|--stop|--health]`。
  配置在项目根 `lightrag.toml`（与 Config.toml/schema.toml 解耦）；密钥走 `*_env`
  间接引用（LIGHTRAG_PG_PASSWORD / LIGHTRAG_NEO4J_PASSWORD / LIGHTRAG_LLM_API_KEY …），
  当前真实值可从 /workspace/semrag/.env 提取注入环境。
  venv 探测顺序含 /workspace/LightRAG/.venv（shebang 断链，脚本用 python -m 启动绕开）。
- 依赖服务：Postgres localhost:5455（semdoc_kernel workspace）、Neo4j localhost:7687。
- 插件协议（src/plugins/graph.rs，lightrag-hku 1.5.6）：
  - insert: `POST /documents/text` `{"text","file_source":"<docid>.txt"}`（异步，不接受 ids）
  - delete: `DELETE /documents/delete_document` `{"doc_ids":["doc-md5(<docid>.txt)", "<docid>"]}`
    （内部 id = md5(file_source)，须同时带上；pipeline busy 时返回 200 + status=busy）
  - query: `POST /query`（LLM 答案）/ `POST /query/data`（结构化）
