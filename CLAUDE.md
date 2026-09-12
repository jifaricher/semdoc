# CLAUDE.md

## 编译规则（必须遵守）

- 所有二进制编译一律使用 `cargo build --release --bins`
- 禁止 `cargo build --release` 和 `cargo build --bins`（会让磁盘 iowait 打满）
- 测试用 `cargo test --lib`；lint 用 `cargo clippy --all-targets`

## 与 lightrag 交互（backend = "lightrag-server"）

- lightrag-server 用 `/workspace/semrag/scripts/start-lightrag-server.sh` 体系启动；
  本机 venv 实际在 `/workspace/LightRAG/.venv`（脚本默认路径 /workspace/web/LightRAG 已失效，
  需 `LIGHTRAG_VENV=/workspace/LightRAG PYTHONPATH=/workspace/LightRAG` 启动，
  或用 `.venv/bin/python -m lightrag.api.lightrag_server`）。
- 依赖服务：Postgres localhost:5455（semdoc_kernel workspace）、Neo4j localhost:7687。
- 插件协议（src/plugins/graph.rs，lightrag-hku 1.5.6）：
  - insert: `POST /documents/text` `{"text","file_source":"<docid>.txt"}`（异步，不接受 ids）
  - delete: `DELETE /documents/delete_document` `{"doc_ids":["doc-md5(<docid>.txt)", "<docid>"]}`
    （内部 id = md5(file_source)，须同时带上；pipeline busy 时返回 200 + status=busy）
  - query: `POST /query`（LLM 答案）/ `POST /query/data`（结构化）
