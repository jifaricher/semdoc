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

/// Webhook sink configured by the host binary (semdoc-server reads
/// [server] from Config.toml and registers it at startup; stdio/CLI have
/// no webhook config and skip registration).
static WEBHOOK: std::sync::OnceLock<crate::config::ServerConfig> = std::sync::OnceLock::new();

pub fn register_webhooks(cfg: crate::config::ServerConfig) {
    let _ = WEBHOOK.set(cfg);
}

/// Fire-and-forget event delivery. Never fails the caller; malformed
/// config or unreachable receivers only log. Payload is HMAC-signed when
/// a secret is configured so receivers can authenticate the source.
/// `X-Semdoc-Signature` value: `sha256=<hex>` HMAC of the exact request
/// body. Exposed as a pure function so the wire format is testable.
pub fn sign_body(body: &str, secret: &str) -> Option<String> {
    if secret.is_empty() {
        return None;
    }
    use hmac::{Hmac, Mac};
    let mut mac = Hmac::<sha2::Sha256>::new_from_slice(secret.as_bytes())
        .expect("hmac accepts any key length");
    mac.update(body.as_bytes());
    Some(format!("sha256={:x}", mac.finalize().into_bytes()))
}

pub fn fire_event(event: &str, doc_id: &str, extra: serde_json::Value) {
    let Some(cfg) = WEBHOOK.get() else { return };
    if !cfg.webhook_enabled(event) {
        return;
    }
    let Some(url) = cfg.webhook_url.clone() else { return };
    let secret = cfg.webhook_secret();
    let event = event.to_string();
    let doc_id = doc_id.to_string();
    tokio::spawn(async move {
        let body = serde_json::json!({
            "event": event,
            "doc_id": doc_id,
            "ts": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            "data": extra,
        });
        let body = match serde_json::to_string(&body) {
            Ok(b) => b,
            Err(_) => return,
        };
        let client = match crate::tls::apply_async(reqwest::Client::builder())
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[webhook] client build failed: {e}");
                return;
            }
        };
        let mut req = client.post(&url).header("content-type", "application/json");
        if let Some(sig) = sign_body(&body, secret.as_deref().unwrap_or("")) {
            req = req.header("X-Semdoc-Signature", sig);
        }
        match req.body(body).send().await {
            Ok(r) if r.status().is_success() => {}
            Ok(r) => eprintln!("[webhook] {event} → {url}: HTTP {}", r.status()),
            Err(e) => eprintln!("[webhook] {event} → {url}: {e}"),
        }
    });
}

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
                "this database declares no [fields] — nothing is updatable".to_string()
            } else {
                format!("available fields and types: {}", parts.join(", "))
            }
        };
        json!({
            "tools": [
                {
                    "name": "query_semantic",
                    "description": "Semantic search: vector ANN runs over leaf chunks (~512 chars), then expands per expand_to (small-to-big). Returns the matched chunks themselves by default (precision-first); for whole documents use parent_doc_id + get_document.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Query text (natural language)" },
                            "limit": { "type": "integer", "default": 10, "description": "Max results to return, up to 200" },
                            "expand_to": { "type": "string", "enum": ["chunk", "parent", "auto"], "default": "chunk", "description": "chunk=return only the matched chunks; parent=return the whole parent document of each hit (deduplicated by parent id); auto=return the parent document when >=2 chunks of the same parent hit, otherwise a merged window of the hit plus/minus 1 adjacent chunks" },
                            "filter": { "type": "object", "description": format!("Mongo-style filter applied before vector search (ANN pre-filter). Schema fields in this database: {filter_desc}. Supported operators: $eq,$ne,$in,$nin,$gt,$gte,$lt,$lte,$exists,$and,$or.") }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_text",
                    "description": "Full-text search (BM25 FTS): keyword matching over raw_text. Best for exact terms, function names, error codes and other lexical recall; complements semantic search.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Query keywords/phrase" },
                            "limit": { "type": "integer", "default": 10 },
                            "filter": { "type": "object", "description": "Mongo-style filter (same as query_semantic)" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_reranked",
                    "description": "Semantic search + cross-encoder reranking: ANN first recalls 80 leaf chunks, then a reranker (e.g. bge-reranker-v2-m3) scores each query-doc pair and reorders them. Highest precision, slowest. Falls back to plain semantic order when --rerank is not enabled or the reranker fails.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Query text" },
                            "limit": { "type": "integer", "default": 10 },
                            "recall_k": { "type": "integer", "default": 80, "description": "Candidate pool size sent to the reranker (>= limit). Smaller values cut rerank latency roughly linearly (~20s+ for 80 candidates, ~3s for 10) at a slight recall cost" },
                            "filter": { "type": "object", "description": "Mongo-style filter (same as query_semantic)" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_graph",
                    "description": "Graph retrieval (lightrag backend). Default synthesize=false: returns structured graph data (entities/relations/source chunks — the raw output of lightrag /query/data) plus local documents mapped from graph chunks back to this database (fast, ~9s, no answer generation). synthesize=true: the LLM synthesizes a natural-language answer from the graph (slow, 27-60s uncached, growing with answer length; a repeated identical query hits the lightrag cache at ~9s). Automatically degrades to semantic search when the graph backend is unavailable (response carries degraded:true).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Query text" },
                            "limit": { "type": "integer", "default": 10 },
                            "synthesize": { "type": "boolean", "description": "false (default): return structured graph data + mapped documents (fast; the agent should parse entities/relationships/documents itself). true: additionally call the LLM for a natural-language answer (slow, 27-60s; better for direct human reading; repeated questions hit the cache at ~9s)" },
                            "chunk_top_k": { "type": "integer", "description": "Graph layer: chunks recalled by lightrag (default 20); smaller is faster" },
                            "max_entity_tokens": { "type": "integer", "description": "Graph layer: entity context token budget (default 6000)" },
                            "max_relation_tokens": { "type": "integer", "description": "Graph layer: relation context token budget (default 8000)" },
                            "max_total_tokens": { "type": "integer", "description": "Graph layer: total context token budget (default 30000)" },
                            "filter": { "type": "object", "description": "Mongo-style filter; in data mode it applies to the mapped local documents, in degraded mode to the semantic search" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_hybrid",
                    "description": "Hybrid retrieval: semantic (reranked) and graph queries run in parallel, merged and deduplicated by document id — graph results first (macro-level associations), semantic results after (supporting details). With synthesize=true the graph side additionally calls the LLM for a natural-language answer (slow). Best for questions needing both multi-hop graph relations and source-text evidence.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Query text" },
                            "limit": { "type": "integer", "default": 10 },
                            "synthesize": { "type": "boolean", "description": "Graph side: false (default) = structured graph data (fast); true = LLM-synthesized answer (slow, 27-60s)" },
                            "chunk_top_k": { "type": "integer", "description": "Graph layer: chunks recalled by lightrag (default 20)" },
                            "max_entity_tokens": { "type": "integer", "description": "Graph layer: entity token budget (default 6000)" },
                            "max_relation_tokens": { "type": "integer", "description": "Graph layer: relation token budget (default 8000)" },
                            "max_total_tokens": { "type": "integer", "description": "Graph layer: total token budget (default 30000)" },
                            "filter": { "type": "object", "description": "Mongo-style filter (applies to the semantic side)" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "get_document",
                    "description": "Fetch one record by id (leaf chunk or parent document) and return its full raw_text (query results truncate raw_text at 2000 chars; this tool does not). Pass a leaf chunk's parent_doc_id to get its whole parent document.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "Document id (from query results or the add_document return value)" }
                        },
                        "required": ["id"]
                    }
                },
                {
                    "name": "add_document",
                    "description": "Add a document: automatically chunked (parent + leaf chunks), vectorized (auto_embed vector columns declared in the schema) and stored. Not written to the lightrag graph by default; mirrored only when graph=true and the database has a graph plugin configured (entity extraction is async; failures only warn and do not affect ingestion). Returns the document id (blake3 of the full text). Re-adding identical text is an idempotent upsert: the same id overwrites the old parent+chunks and re-embeds automatically — also the correct way to refresh vectors after editing content.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Document body (markdown or plain text)" },
                            "source_path": { "type": "string", "description": "Source path label (e.g. file path); strongly recommended for traceability" },
                            "source_type": { "type": "string", "description": "Source kind (file/text/directory/glob etc., default text)" },
                            "language": { "type": "string", "description": "Language hint (markdown is split by headings)" },
                            "metadata": { "type": "object", "description": format!("Schema field values (key-value pairs). Fields in this database: {filter_desc}. Types must match (int64 takes integers, bool takes booleans, list<string> takes arrays).") },
                            "graph": { "type": "boolean", "default": false, "description": "Whether to also mirror into the lightrag graph (requires a graph plugin configured on the database; entity extraction is async, failures only warn). Default false: not written to the graph" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "delete_document",
                    "description": "Delete a document by id with cascade cleanup: the parent document + all leaf chunks in LanceDB; when a graph plugin is configured, the corresponding lightrag document is deleted too (the internal id = md5(file_source) is derived automatically; original ids are also accepted. A busy lightrag pipeline returns an error — retry later). Pass a leaf chunk's parent_doc_id to delete the whole document.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "Document id (prefer the parent document id / parent_doc_id)" }
                        },
                        "required": ["id"]
                    }
                },
                {
                    "name": "update_document_metadata",
                    "description": format!("Patch a document's metadata fields by id (in-place update; raw_text/vector columns/chunk structure untouched, zero re-embedding cost). Only scalar fields declared in this database's [fields] are accepted and type-checked; omitted fields stay unchanged; a value of null or an empty array clears the field. cascade=true (default) applies to the parent document and all its leaf chunks. Available fields: {updatable_desc}. Note: do NOT use this tool to change body text — for content changes call add_document with the new text (same-id idempotent upsert, automatic re-embedding)."),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string", "description": "Document id (parent document or leaf chunk)" },
                            "metadata": { "type": "object", "description": "Field → new-value pairs to update (must be schema-declared fields; a null value clears the field)" },
                            "cascade": { "type": "boolean", "description": "true (default): also update all leaf chunks with parent_doc_id = id; false: update only this row" }
                        },
                        "required": ["id", "metadata"]
                    }
                },
                {
                    "name": "list_documents",
                    "description": "Page through the parent documents in the database (chunk_level=0); each entry includes a 200-char raw_text preview. Use it to survey the database and discover document ids to delete or patch.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": { "type": "integer", "default": 20, "description": "Page size, max 200" },
                            "offset": { "type": "integer", "default": 0, "description": "Offset (use with limit for paging)" }
                        }
                    }
                },
                {
                    "name": "stats",
                    "description": "Knowledge-base statistics: total row count (parent documents + leaf chunks).",
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
                let recall_k = args.get("recall_k").and_then(|v| v.as_u64()).map(|v| v as usize);
                server.engine.query_reranked(text, limit, filter.as_deref(), recall_k).await
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

    // Graph mirror is opt-in: only when the caller passes graph=true
    // (best-effort; LanceDB is truth).
    let mirror_graph = args
        .get("graph")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mut graph_note = String::new();
    if mirror_graph && server.graph.name() == "none" {
        graph_note = "（库未配置图谱插件，跳过图谱镜像）".to_string();
    } else if mirror_graph {
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
    fire_event(
        "add",
        &doc_id,
        json!({ "chunks": leaves.len(), "replaced": replaced }),
    );
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
    fire_event("delete", &target, json!({}));
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
        Ok(n) => {
            fire_event("update", id, json!({ "rows": n, "cascade": cascade }));
            json!({
                "content": [{ "type": "text", "text": format!("已更新 {n} 行（id={id}, cascade={cascade}）") }]
            })
        }
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
    let want_answer = args.get("synthesize").and_then(|v| v.as_bool()).unwrap_or(false);
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
                        "degraded": true, "synthesized": Value::Null, "graph": Value::Null, "documents": vals,
                    })).unwrap_or_default() }] })
                }
                Err(e) => err(format!("query failed: {e:#}")),
            }
        }
        Ok(answer) if want_answer => {
            // lightrag /query wraps the answer in {"response": "..."} — unwrap
            // so the agent gets the plain synthesized text.
            let text = serde_json::from_str::<Value>(&answer.content)
                .ok()
                .and_then(|v| v.get("response").and_then(|r| r.as_str()).map(String::from))
                .unwrap_or(answer.content);
            json!({ "content": [{ "type": "text", "text": text }] })
        }
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
    let want_answer = args.get("synthesize").and_then(|v| v.as_bool()).unwrap_or(false);
    let mode = if want_answer { GraphMode::Hybrid } else { GraphMode::Data };
    let n = |k: &str| args.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    let params = GraphQueryParams {
        chunk_top_k: n("chunk_top_k"),
        max_entity_tokens: n("max_entity_tokens"),
        max_relation_tokens: n("max_relation_tokens"),
        max_total_tokens: n("max_total_tokens"),
    };

    let atomic_fut = server.engine.query_reranked(text, limit, filter_sql, None);
    let graph_fut = server.graph.query(text, mode, limit, &params);
    let (atomic_res, graph_res) = tokio::join!(atomic_fut, graph_fut);

    let mut merged: Vec<Value> = Vec::new();
    let mut degraded = false;
    let mut seen = std::collections::HashSet::new();
    let mut synthesized = Value::Null;

    match graph_res {
        Err(e) => {
            eprintln!("[hybrid] graph side degraded: {e:#}");
            degraded = true;
        }
        Ok(a) if want_answer => {
            synthesized = Value::String(a.content);
            merged.push(json!({ "synthesized": synthesized }));
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
        "degraded": degraded, "synthesized": synthesized, "documents": merged,
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
            "instructions": "semdoc knowledge-base MCP server. query_semantic: semantic search (small-to-big); query_text: full-text search; query_reranked: reranked search (requires --rerank); query_graph: graph retrieval (degrades to semantic when lightrag is unavailable); query_hybrid: hybrid retrieval; get_document: fetch a full document; add_document: ingest; delete_document: delete (cascades to the graph); update_document_metadata: edit metadata without re-embedding; list_documents: paged browsing."
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_body_deterministic_and_prefixed() {
        let body = r#"{"event":"add","doc_id":"abc"}"#;
        let a = sign_body(body, "secret").unwrap();
        let b = sign_body(body, "secret").unwrap();
        assert_eq!(a, b, "same body+key must produce identical signature");
        assert!(a.starts_with("sha256="), "signature must carry the scheme prefix");
        assert_ne!(sign_body(body, "other-key").unwrap(), a, "different key must differ");
    }

    #[test]
    fn sign_body_matches_reference_vector() {
        // RFC 4231-style cross-check with an independently computed value:
        // hmac_sha256(key=b"key", msg=b"The quick brown fox jumps over the lazy dog")
        // = f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8
        let sig = sign_body("The quick brown fox jumps over the lazy dog", "key").unwrap();
        assert_eq!(
            sig,
            "sha256=f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
        );
    }

    #[test]
    fn sign_body_empty_secret_is_none() {
        assert!(sign_body("body", "").is_none());
    }

    #[test]
    fn sign_body_covers_body_changes() {
        let a = sign_body(r#"{"n":1}"#, "k").unwrap();
        let b = sign_body(r#"{"n":2}"#, "k").unwrap();
        assert_ne!(a, b, "tampered body must fail verification");
    }
}
