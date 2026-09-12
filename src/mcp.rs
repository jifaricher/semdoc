// SPDX-License-Identifier: MIT OR Apache-2.0
//! MCP business logic for semdoc — transport-agnostic core.
//!
//! Both binaries share this one source of truth:
//! - `semdoc-mcp` (stdio transport) calls `Server::tools_list` / `dispatch`
//!   from its stdin loop.
//! - `semdoc-server` (streamable HTTP transport) calls the same functions
//!   from the `/mcp` axum handler; session state lives in the server.
//!
//! Tool schemas are schema-driven: filterable and updatable fields are
//! derived from the database's `schema.toml` `[fields]` at runtime.

use anyhow::Result;
use serde_json::{json, Value};

use crate::embedding::Embedder;
use crate::query::{record_json, Engine, ExpandTo};
use crate::schema::SchemaConfig;
use crate::store::Store;

pub struct Server {
    pub engine: Engine,
    pub graph: Box<dyn crate::plugins::graph::GraphPlugin>,
}

/// Verify the incoming schema against `<db>/physical_fields.toml`. Databases
/// without a snapshot (created before compat checking) pass with a warning.
fn check_db_schema_compat(db: &str, config: &SchemaConfig) -> Result<()> {
    let path = format!("{db}/physical_fields.toml");
    if !std::path::Path::new(&path).exists() {
        eprintln!(
            "[compat] {path} missing — database predates schema-compat snapshots; \
             skipping column check (re-init or migrate to enable it)"
        );
        return Ok(());
    }
    let physical = SchemaConfig::load_physical_fields(std::path::Path::new(&path))?;
    config.check_compatible_with(&physical)
}

/// Shared state shape used by both the stdio bin and the HTTP server:
/// one Server owns the engine + graph; hosts keep it in an Arc and call
/// `dispatch`/`tools_list` through it.
pub type SharedServer = std::sync::Arc<Server>;

impl Server {
    pub async fn new(db: &str, with_rerank: bool, deploy_config: Option<&str>) -> Result<Self> {
        let cfg_path = format!("{db}/schema.toml");
        let config = SchemaConfig::load(std::path::Path::new(&cfg_path))?;
        check_db_schema_compat(db, &config)?;
        let vec_fields = config.effective_vector_fields(1024);
        let conn = lancedb::connect(db).execute().await?;
        let store = Store::new_existing(
            conn,
            config.table.name.clone(),
            Store::arrow_schema(&config, &vec_fields),
            vec_fields.clone(),
            config.fields.clone(),
        );
        let deploy = crate::config::DeploymentConfig::load(deploy_config)?;
        deploy.apply_chunk_env();
        let embedder = Embedder::load(&deploy.embedding)?;
        let reranker = if with_rerank {
            Some(std::sync::Arc::new(crate::reranker::Reranker::load(&deploy.rerank)?))
        } else {
            None
        };
        let graph = crate::plugins::graph::build_graph_plugin(&config.plugins.graph).await?;
        Ok(Self {
            engine: Engine { store, embedder, reranker, config },
            graph,
        })
    }

    pub fn tools_list(&self) -> Value {
        let filter_desc = serde_json::to_string_pretty(&{
            let mut m = serde_json::Map::new();
            for (k, fc) in &self.engine.config.fields {
                m.insert(k.clone(), Value::String(fc.r#type.clone()));
            }
            m
        })
        .unwrap_or_default();
        // Schema-driven updatable fields description (only [fields] columns,
        // never raw_text / vector columns).
        let updatable_desc = {
            let mut parts = Vec::new();
            for (k, fc) in &self.engine.config.fields {
                parts.push(format!("{k}({})", fc.r#type));
            }
            if parts.is_empty() {
                "此库 [fields] 为空，无可更新字段".to_string()
            } else {
                format!("可用字段及类型: {}", parts.join(", "))
            }
        };
        json!({
            "tools": [
                {
                    "name": "query_semantic",
                    "description": "语义检索：向量 ANN 在叶子 chunk（约512字符）上执行，再按 expand_to 展开（small-to-big）。默认返回命中的 chunk 本身，precision 优先；需要整篇文档时用 parent_doc_id + get_document。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "查询文本（自然语言）" },
                            "limit": { "type": "integer", "default": 10, "description": "返回条数上限，最大 200" },
                            "expand_to": { "type": "string", "enum": ["chunk", "parent", "auto"], "default": "chunk", "description": "chunk=只返回命中 chunk；parent=命中所在的整篇父文档（按父 id 去重）；auto=同一父文档命中≥2个 chunk 时返回父文档，否则返回命中±1相邻 chunk 的合并窗口" },
                            "filter": { "type": "object", "description": format!("Mongo 风格过滤，作用于向量检索之前（ANN pre-filter）。本库 schema 字段: {filter_desc}。支持算子: $eq,$ne,$in,$nin,$gt,$gte,$lt,$lte,$exists,$and,$or。") }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_text",
                    "description": "全文检索（BM25 FTS）：对 raw_text 做关键词匹配，适合精确术语、函数名、错误码等词法召回，与语义检索互补。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "查询关键词/短语" },
                            "limit": { "type": "integer", "default": 10 },
                            "filter": { "type": "object", "description": "Mongo 风格过滤（同 query_semantic）" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_reranked",
                    "description": "语义检索 + cross-encoder 精排：先 ANN 过召回 80 条叶子 chunk，再用 reranker（bge-reranker-v2-m3 等）对 query-doc 对逐一打分重排。精度最高但最慢；未启用 --rerank 或 reranker 故障时自动降级为普通语义序。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "查询文本" },
                            "limit": { "type": "integer", "default": 10 },
                            "filter": { "type": "object", "description": "Mongo 风格过滤（同 query_semantic）" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_graph",
                    "description": "图谱检索（lightrag 后端）。默认 answer=false：返回结构化图谱数据（实体/关系/来源 chunk）+ 由图谱 chunk 映射回本库的本地文档（快，无 LLM）。answer=true：返回 LLM 综合答案（慢，30-60s+，走 LLM）。图谱后端不可用时自动降级为语义检索（响应含 degraded:true）。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "查询文本" },
                            "limit": { "type": "integer", "default": 10 },
                            "answer": { "type": "boolean", "description": "false（默认）：结构化图谱数据 + 映射文档（快）；true：LLM 综合答案（慢）" },
                            "chunk_top_k": { "type": "integer", "description": "图谱层：lightrag 召回的 chunk 数（默认 20），调小更快" },
                            "max_entity_tokens": { "type": "integer", "description": "图谱层：实体上下文 token 预算（默认 6000）" },
                            "max_relation_tokens": { "type": "integer", "description": "图谱层：关系上下文 token 预算（默认 8000）" },
                            "max_total_tokens": { "type": "integer", "description": "图谱层：总上下文 token 预算（默认 30000）" },
                            "filter": { "type": "object", "description": "Mongo 风格过滤；data 模式作用于映射出的本地文档，降级模式作用于语义检索" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_hybrid",
                    "description": "混合检索：语义（reranked）与图谱并行查询，按文档 id 合并去重——图谱结果在前（宏观关联），语义结果在后（佐证细节）。answer=true 时图谱侧返回 LLM 答案。适合需要'既有图谱多跳关系、又有原文依据'的问题。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "查询文本" },
                            "limit": { "type": "integer", "default": 10 },
                            "answer": { "type": "boolean", "description": "图谱侧：false（默认）=结构化图谱数据；true=LLM 综合答案" },
                            "chunk_top_k": { "type": "integer", "description": "图谱层：lightrag 召回 chunk 数（默认 20）" },
                            "max_entity_tokens": { "type": "integer", "description": "图谱层：实体 token 预算（默认 6000）" },
                            "max_relation_tokens": { "type": "integer", "description": "图谱层：关系 token 预算（默认 8000）" },
                            "max_total_tokens": { "type": "integer", "description": "图谱层：总 token 预算（默认 30000）" },
                            "filter": { "type": "object", "description": "Mongo 风格过滤（作用于语义侧）" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "get_document",
                    "description": "按 id 精查一条记录（叶子 chunk 或父文档），返回完整 raw_text（查询结果的 raw_text 截断在 2000 字符，此工具不截断）。传叶子 chunk 的 parent_doc_id 可获取其整篇父文档。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "文档 id（来自查询结果或 add_document 返回值）" }
                        },
                        "required": ["id"]
                    }
                },
                {
                    "name": "add_document",
                    "description": "写入一篇文档：自动切块（parent + 叶子 chunk）、自动向量化（schema 声明的 auto_embed 向量列）并入库；若库配置了图谱插件则同步镜像到 lightrag（实体抽取异步进行，失败只告警不影响入库）。返回文档 id（blake3(全文)）。重复添加相同文本为幂等 upsert：同 id 覆盖旧 parent+chunks 并自动重嵌入——也是修改正文后刷新向量的正确方式。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "文档正文（markdown 或纯文本）" },
                            "source_path": { "type": "string", "description": "来源路径标签（如文件路径），建议必填以便追溯" },
                            "source_type": { "type": "string", "description": "来源形态（file/text/directory/glob 等，默认 text）" },
                            "language": { "type": "string", "description": "语言提示（markdown 时按标题切分）" },
                            "metadata": { "type": "object", "description": format!("schema 字段值（键值对）。本库字段: {filter_desc}。类型须匹配（int64 传整数、bool 传布尔、list<string> 传数组）。") }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "delete_document",
                    "description": "按 id 删除文档，级联清理：LanceDB 中该父文档 + 全部叶子 chunk；若配置了图谱插件则同步删除 lightrag 中的对应文档（内部 id=md5(file_source) 自动推导，同时兼容原始 id；lightrag pipeline busy 时会报错，可稍后重试）。传叶子 chunk 的 parent_doc_id 即可删整篇。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "文档 id（优先传父文档 id / parent_doc_id）" }
                        },
                        "required": ["id"]
                    }
                },
                {
                    "name": "update_document_metadata",
                    "description": format!("按 id 修补文档的元数据字段（原位 update，不修改 raw_text/向量列/chunk 结构，零重嵌入成本）。只接受本库 [fields] 声明的标量字段并校验类型，未提供的字段保持不变；值为 null 或空数组表示清空该字段。cascade=true（默认）时对该父文档及其全部叶子 chunk 同时生效。本库可用字段: {updatable_desc}。注意：需要修改正文时不要用本工具——正文变更请用相同文本调用 add_document（同 id 幂等 upsert，自动重嵌入）。"),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "文档 id（父文档或叶子 chunk 均可）" },
                            "metadata": { "type": "object", "description": "要更新的字段→新值键值对（须为 schema 声明的字段；值为 null 表示清空该字段）" },
                            "cascade": { "type": "boolean", "description": "true（默认）：同时更新 parent_doc_id = id 的所有叶子 chunk；false：只更新该行" }
                        },
                        "required": ["id", "metadata"]
                    }
                },
                {
                    "name": "list_documents",
                    "description": "分页浏览库中的父文档（chunk_level=0），每条含 raw_text 前 200 字符预览。用于了解库内容、发现待删除/修补的文档 id。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": { "type": "integer", "default": 20, "description": "每页条数，最大 200" },
                            "offset": { "type": "integer", "default": 0, "description": "偏移量（配合 limit 翻页）" }
                        }
                    }
                },
                {
                    "name": "stats",
                    "description": "知识库统计：总行数（父文档+叶子 chunk）。",
                    "inputSchema": { "type": "object", "properties": {} }
                }
            ]
        })
    }
}

fn compile_filter(server: &Server, args: &Value) -> Result<Option<String>> {
    match args.get("filter") {
        Some(f) if !f.is_null() && f.as_object().is_some_and(|o| !o.is_empty()) => {
            let pred = crate::query::Pred::from_json(f)?;
            Ok(Some(pred.compile(&server.engine.config)?))
        }
        _ => Ok(None),
    }
}

fn err(msg: impl Into<String>) -> Value {
    json!({ "content": [{ "type": "text", "text": msg.into() }], "isError": true })
}

pub async fn dispatch(server: &Server, name: &str, args: &Value) -> Value {
    match name {
        "query_semantic" | "query_text" | "query_reranked" | "query_graph" | "query_hybrid" => {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if text.is_empty() {
                return err("text is required");
            }
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10).min(crate::query::MAX_LIMIT as u64) as usize;
            let filter = match compile_filter(server, args) {
                Ok(f) => f,
                Err(e) => return err(format!("invalid filter: {e:#}")),
            };
            if name == "query_graph" {
                return query_graph_via_plugin(server, text, limit, filter.as_deref(), args).await;
            }
            if name == "query_hybrid" {
                return query_hybrid_via_plugin(server, text, limit, filter.as_deref(), args).await;
            }
            let res = if name == "query_text" {
                server.engine.query_fts(text, limit, filter.as_deref()).await
            } else if name == "query_reranked" {
                server.engine.query_reranked(text, limit, filter.as_deref()).await
            } else {
                let expand_to = args
                    .get("expand_to")
                    .and_then(|v| v.as_str())
                    .map(ExpandTo::parse)
                    .unwrap_or(ExpandTo::Chunk);
                server
                    .engine
                    .query_semantic(text, limit, filter.as_deref(), expand_to)
                    .await
            };
            match res {
                Ok(docs) => {
                    let vals: Vec<Value> = docs.iter().map(record_json).collect();
                    json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&vals).unwrap_or_default() }] })
                }
                Err(e) => err(format!("query failed: {e:#}")),
            }
        }
        "get_document" => {
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
            if id.is_empty() {
                return err("id is required");
            }
            match server.engine.store.get_by_id(id).await {
                Ok(Some(rec)) => {
                    let mut obj = serde_json::Map::new();
                    obj.insert("id".into(), Value::String(rec.id.clone()));
                    obj.insert("raw_text".into(), Value::String(rec.raw_text.clone()));
                    obj.insert("chunk_level".into(), rec.chunk_level.into());
                    obj.insert("chunk_index".into(), rec.chunk_index.into());
                    obj.insert(
                        "parent_doc_id".into(),
                        rec.parent_doc_id.clone().map(Value::String).unwrap_or(Value::Null),
                    );
                    obj.insert("source_path".into(), Value::String(rec.source_path.clone()));
                    obj.insert("source_type".into(), Value::String(rec.source_type.clone()));
                    for (k, v) in &rec.extra {
                        obj.insert(k.clone(), v.clone());
                    }
                    json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&Value::Object(obj)).unwrap_or_default() }] })
                }
                Ok(None) => err(format!("document not found: {id}")),
                Err(e) => err(format!("get failed: {e:#}")),
            }
        }
        "stats" => match server.engine.store.count().await {
            Ok(n) => json!({ "content": [{ "type": "text", "text": format!("rows: {n}") }] }),
            Err(e) => err(format!("stats failed: {e:#}")),
        },
        "add_document" => add_document(server, args).await,
        "delete_document" => delete_document(server, args).await,
        "update_document_metadata" => update_document_metadata(server, args).await,
        "list_documents" => list_documents(server, args).await,
        other => err(format!("unknown tool: {other}")),
    }
}

/// Add a document: chunk + auto-embed + upsert via the shared write path,
/// then mirror into the graph KB (degrade on failure — LanceDB is truth).
async fn add_document(server: &Server, args: &Value) -> Value {
    use std::collections::HashMap;

    let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
    if text.is_empty() {
        return err("text is required");
    }
    let source_path = args
        .get("source_path")
        .and_then(|v| v.as_str())
        .unwrap_or("inline");
    let source_type = args
        .get("source_type")
        .and_then(|v| v.as_str())
        .unwrap_or("text")
        .to_string();
    let language = args.get("language").and_then(|v| v.as_str()).map(String::from);

    // Metadata: keys must be schema [fields]; values type-checked per field.
    let mut extra = serde_json::Map::new();
    if let Some(meta) = args.get("metadata").and_then(|v| v.as_object()) {
        for (k, v) in meta {
            let Some(fc) = server.engine.config.fields.get(k) else {
                return err(format!(
                    "unknown metadata field `{k}` — not declared in schema [fields]"
                ));
            };
            if let Err(e) = validate_field_value(&fc.r#type, v) {
                return err(format!("metadata field `{k}`: {e}"));
            }
            extra.insert(k.clone(), v.clone());
        }
    }

    let doc_id = blake3::hash(text.as_bytes()).to_hex().to_string();
    let detect_language = || {
        if language.is_some() {
            return language.clone();
        }
        let is_md = source_path.ends_with(".md") || source_path.ends_with(".markdown");
        Some(if is_md { "markdown".to_string() } else { "plain".to_string() })
    };
    // Feed source_path/source_type into extra when the schema declares them
    // (they are ordinary [fields] now; the Record-level fields exist for
    // backward compatibility with the fixed columns of older databases).
    if server.engine.config.fields.contains_key("source_path") {
        extra.entry("source_path".to_string())
            .or_insert_with(|| Value::String(source_path.to_string()));
    }
    if server.engine.config.fields.contains_key("source_type") {
        extra.entry("source_type".to_string())
            .or_insert_with(|| Value::String(source_type.clone()));
    }
    // Replace-on-add: when the schema flags replace_key fields and every one
    // of them is present in this write, delete the matching old version
    // (parent + leaf chunks, lightrag mirror) before writing.
    let mut replaced: Option<String> = None;
    let rk = server.engine.config.replace_key_fields();
    if !rk.is_empty() {
        let mut conds = Vec::new();
        let mut all_present = true;
        for k in &rk {
            match extra.get(*k) {
                Some(v) => {
                    let s = v.as_str().unwrap_or_default().replace('\'', "''");
                    conds.push(format!("{k} = '{s}'"));
                }
                None => {
                    all_present = false;
                    break;
                }
            }
        }
        if all_present {
            let sql = conds.join(" AND ");
            match server.engine.store.find_parents_by(&sql).await {
                Ok(old) => {
                    for p in old {
                        if p.id == doc_id {
                            continue;
                        }
                        if let Err(e) = server.engine.store.delete_by_id(&p.id).await {
                            eprintln!("[add] replace: failed to delete old {}: {e:#}", p.id);
                            continue;
                        }
                        if server.graph.name() != "none" {
                            if let Err(e) = server.graph.delete_with_retry(&p.id, 3).await {
                                eprintln!(
                                    "[add] replace: graph delete degraded for {}: {e:#}",
                                    p.id
                                );
                            }
                        }
                        replaced = Some(p.id);
                    }
                }
                Err(e) => eprintln!("[add] replace lookup failed: {e:#}"),
            }
        }
    }
    let doc = crate::store::InputDoc {
        id: doc_id.clone(),
        raw_text: text.to_string(),
        source_path: source_path.to_string(),
        source_type,
        language: detect_language(),
        extra,
        vectors: Default::default(),
    };

    // Reuse the CLI write path (chunk + embed all auto_embed columns + upsert).
    // It lives in the semdoc bin helpers; replicate inline via write_doc if
    // exposed, otherwise use the same sequence as `semdoc add`.
    let embedder = &server.engine.embedder;
    let mut parent_vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut leaf_vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let leaves_src = crate::chunker::chunk_for(
        &doc.raw_text,
        &doc.id,
        doc.language.as_deref(),
        &doc.source_path,
    );

    for vf in &server.engine.store.vector_fields {
        if !vf.auto_embed {
            continue; // pre-embedded only — must come in doc.vectors
        }
        let src_text = if vf.source == "raw_text" {
            doc.raw_text.clone()
        } else {
            doc.extra
                .get(&vf.source)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string()
        };
        if src_text.is_empty() {
            return err(format!(
                "vector column `{}`: source text field `{}` is empty",
                vf.name, vf.source
            ));
        }
        let parent_vec = match embedder.encode_blocking(src_text).await {
            Ok(v) => v,
            Err(e) => return err(format!("embed failed: {e:#}")),
        };
        if parent_vec.len() != vf.dim {
            return err(format!(
                "vector column `{}`: embedder dim {} != configured dim {}",
                vf.name,
                parent_vec.len(),
                vf.dim
            ));
        }
        parent_vectors.insert(vf.name.clone(), parent_vec);
        let mut lvs: Vec<Vec<f32>> = Vec::with_capacity(leaves_src.len());
        for c in &leaves_src {
            match embedder.encode_blocking(c.text.clone()).await {
                Ok(v) if v.len() == vf.dim => lvs.push(v),
                Ok(v) => {
                    return err(format!(
                        "vector column `{}`: embedder dim {} != configured dim {}",
                        vf.name,
                        v.len(),
                        vf.dim
                    ))
                }
                Err(e) => return err(format!("embed failed: {e:#}")),
            }
        }
        for (idx, lv) in lvs.iter().enumerate() {
            if idx == 0 {
                leaf_vectors.insert(vf.name.clone(), lv.clone());
            } else {
                leaf_vectors.insert(format!("{}#{}", vf.name, idx), lv.clone());
            }
        }
    }

    let parent = crate::store::Record {
        id: doc.id.clone(),
        raw_text: doc.raw_text.clone(),
        chunk_level: 0,
        chunk_index: 0,
        parent_doc_id: None,
        source_path: doc.source_path.clone(),
        source_type: doc.source_type.clone(),
        extra: doc.extra.clone(),
    };
    let leaves: Vec<crate::store::Record> = leaves_src
        .iter()
        .map(|c| crate::store::Record {
            id: c.chunk_id.clone(),
            raw_text: c.text.clone(),
            chunk_level: 1,
            chunk_index: c.chunk_index,
            parent_doc_id: Some(doc.id.clone()),
            source_path: doc.source_path.clone(),
            source_type: doc.source_type.clone(),
            extra: doc.extra.clone(),
        })
        .collect();
    // Per-leaf vector maps keyed by leaf id.
    let mut leaf_vector_maps: HashMap<String, HashMap<String, Vec<f32>>> = HashMap::new();
    for (i, c) in leaves_src.iter().enumerate() {
        let mut m: HashMap<String, Vec<f32>> = HashMap::new();
        for vf in &server.engine.store.vector_fields {
            if !vf.auto_embed {
                continue;
            }
            let key = if i == 0 {
                vf.name.clone()
            } else {
                format!("{}#{}", vf.name, i)
            };
            if let Some(v) = leaf_vectors.get(&key) {
                m.insert(vf.name.clone(), v.clone());
            }
        }
        leaf_vector_maps.insert(c.chunk_id.clone(), m);
    }
    let empty_vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let rows: Vec<(&crate::store::Record, &HashMap<String, Vec<f32>>)> =
        std::iter::once((&parent, &parent_vectors))
            .chain(
                leaves
                    .iter()
                    .map(|l| (l, leaf_vector_maps.get(&l.id).unwrap_or(&empty_vectors))),
            )
            .collect();
    let batch = match server.engine.store.build_batch_public(&rows) {
        Ok(b) => b,
        Err(e) => return err(format!("batch build failed: {e:#}")),
    };
    if let Err(e) = server.engine.store.write_batch(batch).await {
        return err(format!("write failed: {e:#}"));
    }
    if let Err(e) = server.engine.store.optimize_indices().await {
        eprintln!("[mcp] optimize_indices: {e:#}");
    }

    // Graph mirror (best-effort).
    let mut graph_note = String::new();
    if server.graph.name() != "none" {
        match server
            .graph
            .insert(vec![(text.to_string(), doc_id.clone())])
            .await
        {
            Ok(()) => {}
            Err(e) => {
                graph_note = format!("（注意：图谱镜像失败，LanceDB 已写入: {e:#}）");
                eprintln!("[graph] insert degraded: {e:#}");
            }
        }
    }
    json!({
        "content": [{ "type": "text", "text": format!(
            "文档已写入，id: {doc_id}（chunks: {}）{}{graph_note}",
            leaves.len(),
            replaced.as_ref().map(|old| format!("（已按 replace_key 替换旧版本 {old}）")).unwrap_or_default(),
        ) }]
    })
}

/// Delete by id with cascade: LanceDB (parent + leaves via parent_doc_id)
/// then the graph KB. lightrag failure is reported in the response text so
/// MCP clients know the graph side may still hold the doc.
/// Shared delete pipeline used by MCP, REST and CLI: existence check on
/// the parent/leaf id, LanceDB cascade delete, then lightrag mirror delete
/// with busy-retry. Returns a human-readable status line; `Err` when
/// nothing was deleted (unknown id or store failure).
pub async fn delete_doc_checked(server: &Server, id: &str) -> Result<String> {
    if id.is_empty() {
        anyhow::bail!("id is required");
    }
    let Some(existing) = server.engine.store.get_by_id(id).await? else {
        anyhow::bail!("document not found: {id}");
    };
    // Normalize: a leaf chunk id deletes its whole parent document.
    let target = existing.parent_doc_id.clone().unwrap_or_else(|| existing.id.clone());
    server.engine.store.delete_by_id(&target).await?;
    let mut note = String::new();
    if server.graph.name() != "none" {
        match server.graph.delete_with_retry(&target, 3).await {
            Ok(()) => {}
            Err(e) => {
                note = format!(
                    "（注意：图谱侧删除失败（pipeline busy 等），lightrag 可能仍有残留；                     稍后重试同一命令即可补删: {e:#}）"
                );
                eprintln!("[graph] delete degraded: {e:#}");
            }
        }
    }
    Ok(format!("文档 {target} 已删除（chunks 级联清理）{note}"))
}

async fn delete_document(server: &Server, args: &Value) -> Value {
    let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
    if id.is_empty() {
        return err("id is required");
    }
    match delete_doc_checked(server, id).await {
        Ok(msg) => json!({ "content": [{ "type": "text", "text": msg }] }),
        Err(e) => err(format!("{e:#}")),
    }
}

/// Type-check a JSON value against a schema field type (same set as
/// Store::set_scalar accepts).
fn validate_field_value(t: &str, v: &Value) -> anyhow::Result<()> {
    match t {
        "string" | "text" => {
            if !v.is_null() && v.as_str().is_none() {
                anyhow::bail!("expected string");
            }
        }
        "int64" | "timestamp" => {
            if !v.is_null() && v.as_i64().is_none() {
                anyhow::bail!("expected integer");
            }
        }
        "float32" => {
            if !v.is_null() && v.as_f64().is_none() {
                anyhow::bail!("expected number");
            }
        }
        "bool" => {
            if !v.is_null() && v.as_bool().is_none() {
                anyhow::bail!("expected boolean");
            }
        }
        "list<string>" => {
            if !v.is_null() && !v.as_array().is_some_and(|a| a.iter().all(|x| x.is_string())) {
                anyhow::bail!("expected array of strings");
            }
        }
        other => anyhow::bail!("unsupported field type {other}"),
    }
    Ok(())
}

/// Update schema-declared metadata fields in place (no re-embed).
/// `metadata` keys must exist in [fields]; values type-checked; a JSON null
/// clears the field. cascade=true (default) also updates leaf chunks.
async fn update_document_metadata(server: &Server, args: &Value) -> Value {
    let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
    if id.is_empty() {
        return err("id is required");
    }
    let Some(meta) = args.get("metadata").and_then(|v| v.as_object()) else {
        return err("metadata object is required");
    };
    if meta.is_empty() {
        return err("metadata is empty — provide at least one schema field");
    }
    let cascade = args.get("cascade").and_then(|v| v.as_bool()).unwrap_or(true);

    // Validate every key against the schema BEFORE touching the store.
    let mut updates: Vec<(String, Value)> = Vec::new();
    for (k, v) in meta {
        let Some(fc) = server.engine.config.fields.get(k) else {
            return err(format!(
                "unknown metadata field `{k}` — not declared in schema [fields]"
            ));
        };
        if let Err(e) = validate_field_value(&fc.r#type, v) {
            return err(format!("metadata field `{k}`: {e}"));
        }
        updates.push((k.clone(), v.clone()));
    }

    match server
        .engine
        .store
        .update_metadata(id, &updates, cascade)
        .await
    {
        Ok(n) => json!({
            "content": [{ "type": "text", "text": format!("已更新 {n} 行（id={id}, cascade={cascade}）") }]
        }),
        Err(e) => err(format!("update failed: {e:#}")),
    }
}

/// Page through parent documents (chunk_level=0) with a raw_text preview.
async fn list_documents(server: &Server, args: &Value) -> Value {
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(20)
        .min(crate::query::MAX_LIMIT as u64) as usize;
    let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    match server.engine.store.list_parents(limit, offset).await {
        Ok(docs) => {
            let vals: Vec<Value> = docs
                .iter()
                .map(|d| {
                    let mut v = record_json(d);
                    if let Value::Object(ref mut o) = v {
                        let preview: String = d.raw_text.chars().take(200).collect();
                        o.insert("raw_text_preview".into(), Value::String(preview));
                    }
                    v
                })
                .collect();
            json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&vals).unwrap_or_default() }] })
        }
        Err(e) => err(format!("list failed: {e:#}")),
    }
}

/// Graph query via the plugin. Data mode (answer=false, default): map
/// data.chunks[].doc_id back to local parent documents. Hybrid (answer=true):
/// return the synthesized answer text.
async fn query_graph_via_plugin(
    server: &Server,
    text: &str,
    limit: usize,
    filter_sql: Option<&str>,
    args: &Value,
) -> Value {
    use crate::plugins::graph::{GraphMode, GraphQueryParams};
    let want_answer = args.get("answer").and_then(|v| v.as_bool()).unwrap_or(false);
    let mode = if want_answer { GraphMode::Hybrid } else { GraphMode::Data };
    let n = |k: &str| args.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    let params = GraphQueryParams {
        chunk_top_k: n("chunk_top_k"),
        max_entity_tokens: n("max_entity_tokens"),
        max_relation_tokens: n("max_relation_tokens"),
        max_total_tokens: n("max_total_tokens"),
    };
    match server.graph.query(text, mode, limit, &params).await {
        Err(e) => {
            eprintln!("[graph] degraded to semantic: {e:#}");
            match server.engine.query_semantic(text, limit, filter_sql, ExpandTo::Chunk).await {
                Ok(docs) => {
                    let vals: Vec<Value> = docs.iter().map(record_json).collect();
                    json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&json!({
                        "degraded": true, "answer": Value::Null, "graph": Value::Null, "documents": vals,
                    })).unwrap_or_default() }] })
                }
                Err(e) => err(format!("query failed: {e:#}")),
            }
        }
        Ok(answer) if want_answer => json!({ "content": [{ "type": "text", "text": answer.content }] }),
        Ok(answer) => {
            // Data mode: structured envelope + mapped local documents.
            let raw: Value = match serde_json::from_str(&answer.content) {
                Ok(v) => v,
                Err(e) => return err(format!("graph data decode: {e}")),
            };
            let data = raw.get("data").cloned().unwrap_or(Value::Null);
            let chunk_doc_ids: Vec<String> = data
                .get("chunks")
                .and_then(|c| c.as_array())
                .map(|chunks| {
                    chunks
                        .iter()
                        .filter_map(|c| c.get("doc_id"))
                        .filter_map(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect()
                })
                .unwrap_or_default();
            let mut documents: Vec<Value> = Vec::new();
            if !chunk_doc_ids.is_empty() {
                match server.engine.store.get_parents(&chunk_doc_ids).await {
                    Ok(parents) => {
                        for id in &chunk_doc_ids {
                            if let Some(p) = parents.get(id) {
                                documents.push(record_json(p));
                            }
                        }
                    }
                    Err(e) => return err(format!("map documents: {e:#}")),
                }
            }
            // Fallback mapping when chunks lack doc_id (lightrag HTTP builds
            // without the semrag Python wrapper): match chunk content against
            // parent raw_text. Parent counts are small (hundreds).
            if documents.is_empty() {
                let contents: Vec<String> = data
                    .get("chunks")
                    .and_then(|c| c.as_array())
                    .map(|chunks| {
                        chunks
                            .iter()
                            .filter_map(|c| c.get("content"))
                            .filter_map(|v| v.as_str())
                            .filter(|s| s.chars().count() >= 50)
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default();
                if !contents.is_empty() {
                    if let Ok(all) = server.engine.store.all_parents().await {
                        for p in &all {
                            if documents.len() >= limit {
                                break;
                            }
                            if contents.iter().any(|c| p.raw_text.contains(c.as_str())) {
                                documents.push(record_json(p));
                            }
                        }
                    }
                }
            }
            documents.truncate(limit);
            json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&json!({
                "degraded": false, "graph": data, "documents": documents,
            })).unwrap_or_default() }] })
        }
    }
}

/// Hybrid query: atomic (reranked) + graph in parallel, merged by id.
async fn query_hybrid_via_plugin(
    server: &Server,
    text: &str,
    limit: usize,
    filter_sql: Option<&str>,
    args: &Value,
) -> Value {
    use crate::plugins::graph::{GraphMode, GraphQueryParams};
    let want_answer = args.get("answer").and_then(|v| v.as_bool()).unwrap_or(false);
    let mode = if want_answer { GraphMode::Hybrid } else { GraphMode::Data };
    let n = |k: &str| args.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    let params = GraphQueryParams {
        chunk_top_k: n("chunk_top_k"),
        max_entity_tokens: n("max_entity_tokens"),
        max_relation_tokens: n("max_relation_tokens"),
        max_total_tokens: n("max_total_tokens"),
    };

    let atomic_fut = server.engine.query_reranked(text, limit, filter_sql);
    let graph_fut = server.graph.query(text, mode, limit, &params);
    let (atomic_res, graph_res) = tokio::join!(atomic_fut, graph_fut);

    let mut merged: Vec<Value> = Vec::new();
    let mut degraded = false;
    let mut seen = std::collections::HashSet::new();
    let mut answer_json = Value::Null;

    match graph_res {
        Err(e) => {
            eprintln!("[hybrid] graph side degraded: {e:#}");
            degraded = true;
        }
        Ok(a) if want_answer => {
            answer_json = Value::String(a.content);
            merged.push(json!({ "answer": answer_json }));
        }
        Ok(a) => {
            if let Ok(raw) = serde_json::from_str::<Value>(&a.content) {
                let data = raw.get("data").cloned().unwrap_or(Value::Null);
                let chunk_doc_ids: Vec<String> = data
                    .get("chunks")
                    .and_then(|c| c.as_array())
                    .map(|chunks| {
                        chunks
                            .iter()
                            .filter_map(|c| c.get("doc_id"))
                            .filter_map(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect()
                    })
                    .unwrap_or_default();
                if !chunk_doc_ids.is_empty() {
                    if let Ok(parents) = server.engine.store.get_parents(&chunk_doc_ids).await {
                        for id in &chunk_doc_ids {
                            if let Some(p) = parents.get(id) {
                                if seen.insert(p.id.clone()) {
                                    merged.push(record_json(p));
                                }
                            }
                        }
                    }
                }
                if merged.is_empty() {
                    if let Ok(all) = server.engine.store.all_parents().await {
                        let contents: Vec<String> = data
                            .get("chunks")
                            .and_then(|c| c.as_array())
                            .map(|chunks| {
                                chunks
                                    .iter()
                                    .filter_map(|c| c.get("content"))
                                    .filter_map(|v| v.as_str())
                                    .filter(|s| s.chars().count() >= 50)
                                    .map(String::from)
                                    .collect()
                            })
                            .unwrap_or_default();
                        for p in &all {
                            if merged.len() >= limit {
                                break;
                            }
                            if contents.iter().any(|c| p.raw_text.contains(c.as_str()))
                                && seen.insert(p.id.clone())
                            {
                                merged.push(record_json(p));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Ok(atomic) = atomic_res {
        for d in atomic {
            if merged.len() >= limit {
                break;
            }
            if seen.insert(d.id.clone()) {
                merged.push(record_json(&d));
            }
        }
    }

    json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&json!({
        "degraded": degraded, "answer": answer_json, "documents": merged,
    })).unwrap_or_default() }] })
}

/// JSON-RPC method dispatch for the MCP protocol (transport-agnostic).
/// `params` is the request's `params`; the return value is the JSON-RPC
/// `result` (or an object containing an `error` — the transport merges it
/// into the envelope). Tools are executed via [`dispatch`].
pub async fn handle_request(server: &Server, method: &str, params: &Value) -> Value {
    match method {
        "initialize" => json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "semdoc", "version": "0.1.0" },
            "instructions": "semdoc 知识库 MCP 服务。query_semantic 语义检索（small-to-big）；query_text 全文检索；query_reranked 精排（需 --rerank）；query_graph 图谱检索（lightrag 不可用时降级语义）；query_hybrid 混合检索；get_document 精查整篇；add_document 写入；delete_document 删除（级联图谱）；update_document_metadata 改标签不重嵌入；list_documents 分页浏览。"
        }),
        "notifications/initialized" => json!(null),
        "tools/list" => server.tools_list(),
        "tools/call" => {
            let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            dispatch(server, name, &arguments).await
        }
        other => json!({
            "error": { "code": -32601, "message": format!("method not found: {other}") }
        }),
    }
}
