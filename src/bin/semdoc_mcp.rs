// SPDX-License-Identifier: MIT OR Apache-2.0
//! semdoc MCP server (stdio) — tools: query_semantic / query_text /
//! query_reranked / stats / list_documents.
//!
//! Schema-driven: the `filter` input is the Mongo-style JSON DSL, validated
//! against the database's schema at query time.

use anyhow::Result;
use clap::Parser;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

use semdoc::embedding::Embedder;
use semdoc::query::{record_json, Engine, ExpandTo};
use semdoc::schema::SchemaConfig;
use semdoc::store::Store;

#[derive(Parser)]
#[command(name = "semdoc-mcp", about = "MCP server for semdoc")]
struct Args {
    /// Database directory (contains schema.toml + LanceDB files)
    #[arg(short, long)]
    db: String,
    /// Enable reranked queries (loads the reranker backend from config/env)
    #[arg(long, default_value_t = false)]
    rerank: bool,
    /// Deployment config file (default: $SEMDOC_CONFIG or ./semdoc.config.toml)
    #[arg(long)]
    config: Option<String>,
}

pub struct Server {
    pub engine: Engine,
    pub graph: Box<dyn semdoc::plugins::graph::GraphPlugin>,
}

impl Server {
    pub async fn new(db: &str, with_rerank: bool, deploy_config: Option<&str>) -> Result<Self> {
        let cfg_path = format!("{db}/schema.toml");
        let config = SchemaConfig::load(std::path::Path::new(&cfg_path))?;
        let vec_fields = config.effective_vector_fields(1024);
        let conn = lancedb::connect(db).execute().await?;
        let store = Store::new_existing(
            conn,
            config.table.name.clone(),
            Store::arrow_schema(&config, &vec_fields),
            vec_fields.clone(),
            config.fields.clone(),
        );
        let deploy = semdoc::config::DeploymentConfig::load(deploy_config.as_deref())?;
        deploy.apply_chunk_env();
        let embedder = Embedder::load(&deploy.embedding)?;
        let reranker = if with_rerank {
            Some(std::sync::Arc::new(semdoc::reranker::Reranker::load(&deploy.rerank)?))
        } else {
            None
        };
        let graph = semdoc::plugins::graph::build_graph_plugin(&config.plugins.graph).await?;
        Ok(Self {
            engine: Engine { store, embedder, reranker, config },
            graph,
        })
    }

    fn tools_list(&self) -> Value {
        let filter_desc = serde_json::to_string_pretty(&{
            let mut m = serde_json::Map::new();
            for (k, fc) in &self.engine.config.fields {
                m.insert(k.clone(), Value::String(fc.r#type.clone()));
            }
            m
        })
        .unwrap_or_default();
        json!({
            "tools": [
                {
                    "name": "query_semantic",
                    "description": "Semantic search over the knowledge base (small-to-big: ANN over leaf chunks, expand per expand_to).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" },
                            "limit": { "type": "integer", "default": 10 },
                            "expand_to": { "type": "string", "enum": ["chunk", "parent", "auto"], "default": "chunk" },
                            "filter": { "type": "object", "description": format!("Mongo-style filter. Schema fields: {filter_desc}. Ops: $eq,$ne,$in,$nin,$gt,$gte,$lt,$lte,$exists,$and,$or.") }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_text",
                    "description": "Full-text search (FTS).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" },
                            "limit": { "type": "integer", "default": 10 },
                            "filter": { "type": "object" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_reranked",
                    "description": "Semantic search + cross-encoder rerank (leaf-chunk granularity).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" },
                            "limit": { "type": "integer", "default": 10 },
                            "filter": { "type": "object" }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_graph",
                    "description": "Graph retrieval via the graph plugin. Default (answer=false): structured graph data (entities/relationships/chunks) plus the local documents mapped from the graph chunks (fast, no LLM). answer=true: LLM-synthesized answer (slow, 30-60s+). Degrades to semantic search when the graph backend is unavailable.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" },
                            "limit": { "type": "integer", "default": 10 },
                            "answer": { "type": "boolean", "description": "false (default): structured graph data + mapped documents (fast). true: LLM-synthesized answer (slow)." },
                            "chunk_top_k": { "type": "integer", "description": "Graph layer: number of chunks lightrag retrieves (default 20). Lower = smaller/faster." },
                            "max_entity_tokens": { "type": "integer", "description": "Graph layer: entity context token budget (default 6000)." },
                            "max_relation_tokens": { "type": "integer", "description": "Graph layer: relationship context token budget (default 8000)." },
                            "max_total_tokens": { "type": "integer", "description": "Graph layer: total context token budget (default 30000)." },
                            "filter": { "type": "object", "description": "Mongo-style filter; applies to the mapped documents (data mode) or the semantic fallback." }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "query_hybrid",
                    "description": "Parallel atomic (reranked semantic) + graph retrieval, merged and deduped by id. Graph side first (summary), atomic side as supporting evidence. Filter applies to the atomic side.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" },
                            "limit": { "type": "integer", "default": 10 },
                            "answer": { "type": "boolean", "description": "Graph side: false (default) = structured graph data; true = LLM-synthesized answer." },
                            "chunk_top_k": { "type": "integer", "description": "Graph layer: number of chunks lightrag retrieves (default 20)." },
                            "max_entity_tokens": { "type": "integer", "description": "Graph layer: entity token budget (default 6000)." },
                            "max_relation_tokens": { "type": "integer", "description": "Graph layer: relationship token budget (default 8000)." },
                            "max_total_tokens": { "type": "integer", "description": "Graph layer: total token budget (default 30000)." },
                            "filter": { "type": "object", "description": "Mongo-style filter (atomic side)." }
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "stats",
                    "description": "Knowledge base statistics.",
                    "inputSchema": { "type": "object", "properties": {} }
                }
            ]
        })
    }
}

fn compile_filter(server: &Server, args: &Value) -> Result<Option<String>> {
    match args.get("filter") {
        Some(f) if !f.is_null() && f.as_object().is_some_and(|o| !o.is_empty()) => {
            let pred = semdoc::query::Pred::from_json(f)?;
            Ok(Some(pred.compile(&server.engine.config)?))
        }
        _ => Ok(None),
    }
}

fn err(msg: impl Into<String>) -> Value {
    json!({ "content": [{ "type": "text", "text": msg.into() }], "isError": true })
}

async fn dispatch(server: &Server, name: &str, args: &Value) -> Value {
    match name {
        "query_semantic" | "query_text" | "query_reranked" | "query_graph" | "query_hybrid" => {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if text.is_empty() {
                return err("text is required");
            }
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
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
        "stats" => match server.engine.store.count().await {
            Ok(n) => json!({ "content": [{ "type": "text", "text": format!("rows: {n}") }] }),
            Err(e) => err(format!("stats failed: {e:#}")),
        },
        other => err(format!("unknown tool: {other}")),
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
    use semdoc::plugins::graph::{GraphMode, GraphQueryParams};
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
    use semdoc::plugins::graph::{GraphMode, GraphQueryParams};
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

enum ReadResult {
    Eof,
    Empty,
    Msg(Value),
    ParseError(String),
}

fn read_message() -> ReadResult {
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => ReadResult::Eof,
        Ok(_) => {
            if line.trim().is_empty() {
                ReadResult::Empty
            } else {
                match serde_json::from_str(&line) {
                    Ok(v) => ReadResult::Msg(v),
                    Err(e) => ReadResult::ParseError(e.to_string()),
                }
            }
        }
        Err(e) => ReadResult::ParseError(e.to_string()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("semdoc MCP server starting (db: {})", args.db);
    let server = Server::new(&args.db, args.rerank, args.config.as_deref()).await?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    loop {
        match read_message() {
            ReadResult::Eof => break,
            ReadResult::Empty => continue,
            ReadResult::ParseError(m) => {
                let resp = json!({
                    "jsonrpc": "2.0", "id": null,
                    "error": { "code": -32700, "message": format!("parse error: {m}") }
                });
                writeln!(out, "{}", resp)?;
                out.flush()?;
                continue;
            }
            ReadResult::Msg(msg) => {
                let id = msg.get("id").cloned().unwrap_or(Value::Null);
                let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
                let result = match method {
                    "initialize" => json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "semdoc", "version": "0.1.0" }
                    }),
                    "notifications/initialized" => json!(null),
                    "tools/list" => server.tools_list(),
                    "tools/call" => {
                        let params = msg.get("params").cloned().unwrap_or(json!({}));
                        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
                        dispatch(&server, name, &arguments).await
                    }
                    other => json!({
                        "error": { "code": -32601, "message": format!("method not found: {other}") }
                    }),
                };
                let resp = json!({ "jsonrpc": "2.0", "id": id, "result": result });
                writeln!(out, "{}", resp)?;
                out.flush()?;
            }
        }
    }
    Ok(())
}
