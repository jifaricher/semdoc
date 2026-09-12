// SPDX-License-Identifier: MIT OR Apache-2.0
//! semdoc-server — HTTP REST + graph plugin + MCP streamable HTTP endpoints.
//!
//!   POST /documents            add doc (auto-embed)
//!   POST /documents/delete     {id}
//!   POST /query/semantic       {text, limit, filter?, expand_to?}
//!   POST /query/text           {text, limit, filter?}
//!   POST /query/reranked       {text, limit, filter?}
//!   POST /query/graph          {text, limit} — via graph plugin, degrade to semantic
//!   POST /query/hybrid         {text, limit} — parallel atomic+graph, merged
//!   GET  /documents/:id        fetch one row (full raw_text)
//!   GET  /stats
//!   GET  /health
//!
//! MCP streamable HTTP (2025-03-26 spec):
//!   POST   /mcp   JSON-RPC over HTTP, optional SSE response; requires
//!                 Authorization: Bearer <$SEMDOC_MCP_TOKEN> on every call.
//!                 If the env var is unset, /mcp refuses all requests.
//!   DELETE /mcp   terminates the Mcp-Session-Id session.
//! Sessions are idle-TTL'd (24h) and persisted to SQLite so restarts
//! don't kick connected clients.
//!
//! `filter` is the Mongo-style JSON DSL compiled against the schema.

use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use semdoc::mcp::Server as McpServer;
use semdoc::plugins::graph::GraphMode;
use semdoc::query::{record_json, ExpandTo};
use semdoc::store::InputDoc;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "semdoc-server", about = "HTTP server for semdoc")]
struct Args {
    #[arg(short, long)]
    db: String,
    #[arg(short, long, default_value = "127.0.0.1:8092")]
    listen: String,
    /// Enable reranked queries
    #[arg(long, default_value_t = false)]
    rerank: bool,
    /// Require Authorization: Bearer <$token_env> when set (config: server.token_env,
    /// env: SEMDOC_TOKEN as fallback indirection)
    #[arg(long, default_value_t = false)]
    auth: bool,
    /// Deployment config file (default: $SEMDOC_CONFIG or ./semdoc.config.toml)
    #[arg(long)]
    config: Option<String>,
}

struct AppState {
    /// Shared MCP server — owns the Engine (store/embedder/reranker/config)
    /// and the graph plugin. REST handlers reach them via `state.mcp.engine`.
    mcp: McpServer,
    token: Option<String>,
    mcp_sessions: Arc<McpSessionStore>,
    /// Bearer token for /mcp. None = /mcp refuses everything (fail-closed).
    expected_token: Option<Arc<String>>,
}

#[derive(Deserialize)]
struct AddBody {
    text: String,
    #[serde(default)]
    source_path: Option<String>,
    #[serde(default)]
    meta: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct QueryBody {
    text: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    expand_to: Option<String>,
    /// query_graph only: true = LLM-synthesized answer (slow). Default false
    /// = structured graph data (entities/relationships/chunks) + the local
    /// documents mapped from the graph chunks (fast, no LLM).
    #[serde(default)]
    answer: Option<bool>,
    /// Graph-layer knobs passed through to lightrag's QueryParam
    /// (query_graph / query_hybrid; unset = lightrag defaults).
    #[serde(default)]
    chunk_top_k: Option<usize>,
    #[serde(default)]
    max_entity_tokens: Option<usize>,
    #[serde(default)]
    max_relation_tokens: Option<usize>,
    #[serde(default)]
    max_total_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct DeleteBody {
    id: String,
}

fn auth_ok(state: &AppState, headers: &HeaderMap) -> bool {
    match &state.token {
        None => true,
        Some(t) => {
            let h = headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "));
            h == Some(t.as_str())
        }
    }
}

fn filter_sql(state: &AppState, body: &QueryBody) -> Result<Option<String>, String> {
    match &body.filter {
        Some(f) if f.as_object().is_some_and(|o| !o.is_empty()) => {
            let pred = semdoc::query::Pred::from_json(f).map_err(|e| e.to_string())?;
            pred.compile(&state.mcp.engine.config).map(Some).map_err(|e| e.to_string())
        }
        _ => Ok(None),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    // One Server (engine + graph) backs both the REST handlers and /mcp.
    let mcp = McpServer::new(&args.db, args.rerank, args.config.as_deref()).await?;
    let deploy = semdoc::config::DeploymentConfig::load(args.config.as_deref())?;
    deploy.apply_chunk_env();
    let token = deploy
        .server_token()
        .or_else(|| std::env::var("SEMDOC_TOKEN").ok());
    if args.auth && token.is_none() {
        anyhow::bail!("--auth requires server.token_env (config) or SEMDOC_TOKEN (env)");
    }
    // MCP streamable HTTP: session store persisted next to the LanceDB dir
    // (separate SQLite file so restarts rehydrate sessions), plus a
    // background sweeper for idle-TTL expiry.
    let mcp_sessions_path = format!("{}/mcp_sessions.sqlite3", args.db);
    let mcp_sessions = Arc::new(McpSessionStore::open(&mcp_sessions_path));
    {
        let sessions_for_sweep = Arc::clone(&mcp_sessions);
        tokio::spawn(async move {
            sessions_for_sweep.ttl_sweep(MCP_SESSION_TTL, MCP_SWEEP_PERIOD).await;
        });
    }
    let expected_token = std::env::var("SEMDOC_MCP_TOKEN").ok().map(Arc::new);
    if expected_token.is_none() {
        eprintln!(
            "warning: SEMDOC_MCP_TOKEN not set — /mcp endpoint will refuse all \
             requests. Set SEMDOC_MCP_TOKEN=<secret> to enable remote MCP."
        );
    } else {
        eprintln!("/mcp endpoint enabled with bearer-token auth.");
    }

    let state = Arc::new(AppState {
        mcp,
        token,
        mcp_sessions,
        expected_token,
    });

    let app = Router::new()
        .route("/documents", post(add_doc))
        .route("/documents/delete", post(delete_doc))
        .route("/documents/:id", get(get_doc))
        .route("/query/semantic", post(query_semantic))
        .route("/query/text", post(query_text))
        .route("/query/reranked", post(query_reranked))
        .route("/query/graph", post(query_graph))
        .route("/query/hybrid", post(query_hybrid))
        .route("/mcp", post(mcp_post).delete(mcp_delete))
        .route("/stats", get(stats))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    eprintln!("semdoc-server listening on {}", args.listen);
    axum::serve(listener, app).await?;
    Ok(())
}

type Resp<T> = Result<Json<T>, (StatusCode, String)>;

async fn add_doc(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<AddBody>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let source_path = body.source_path.unwrap_or_else(|| "inline".into());
    let mut extra: serde_json::Map<String, Value> = body.meta;
    if state.mcp.engine.config.fields.contains_key("source_path") {
        extra.entry("source_path".to_string())
            .or_insert_with(|| Value::String(source_path.clone()));
    }
    if state.mcp.engine.config.fields.contains_key("source_type") {
        extra.entry("source_type".to_string())
            .or_insert_with(|| Value::String("text".to_string()));
    }
    let doc = InputDoc {
        id: blake3::hash(body.text.as_bytes()).to_hex().to_string(),
        raw_text: body.text.clone(),
        source_path,
        source_type: "text".into(),
        language: Some("markdown".into()),
        extra,
        vectors: Default::default(),
    };
    semdoc_bin_helpers::write_doc(&state.mcp.engine.store, &state.mcp.engine.embedder, doc)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    // Mirror into the graph KB (entity extraction is slow; degrade loudly,
    // never fail the write — LanceDB is the source of truth).
    let graph_text = body.text.clone();
    let graph_id = blake3::hash(body.text.as_bytes()).to_hex().to_string();
    if let Err(e) = state.mcp.graph.insert(vec![(graph_text, graph_id)]).await {
        eprintln!("[graph] insert degraded: {e:#}");
    }
    Ok(Json(json!({"ok": true})))
}

async fn delete_doc(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<DeleteBody>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    state
        .mcp.engine
        .store
        .delete_by_id(&body.id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    if let Err(e) = state.mcp.graph.delete(&body.id).await {
        eprintln!("[graph] delete degraded: {e:#}");
    }
    Ok(Json(json!({"ok": true})))
}

async fn get_doc(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    match state.mcp.engine.store.get_by_id(&id).await {
        Ok(Some(rec)) => {
            let mut obj = serde_json::Map::new();
            obj.insert("id".into(), Value::String(rec.id.clone()));
            // Full raw_text — this endpoint exists to bypass the 2000-char cap.
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
            Ok(Json(Value::Object(obj)))
        }
        Ok(None) => Err((StatusCode::NOT_FOUND, format!("document not found: {id}"))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))),
    }
}

async fn query_semantic(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<QueryBody>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let sql = filter_sql(&state, &body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let expand_to = body.expand_to.as_deref().map(ExpandTo::parse).unwrap_or(ExpandTo::Chunk);
    let docs = state
        .mcp.engine
        .query_semantic(&body.text, body.limit.unwrap_or(10).min(semdoc::query::MAX_LIMIT), sql.as_deref(), expand_to)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(Json(json!({ "documents": docs.iter().map(record_json).collect::<Vec<_>>() })))
}

async fn query_text(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<QueryBody>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let sql = filter_sql(&state, &body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let docs = state
        .mcp.engine
        .query_fts(&body.text, body.limit.unwrap_or(10).min(semdoc::query::MAX_LIMIT), sql.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(Json(json!({ "documents": docs.iter().map(record_json).collect::<Vec<_>>() })))
}

async fn query_reranked(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<QueryBody>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let sql = filter_sql(&state, &body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let docs = state
        .mcp.engine
        .query_reranked(&body.text, body.limit.unwrap_or(10).min(semdoc::query::MAX_LIMIT), sql.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(Json(json!({ "documents": docs.iter().map(record_json).collect::<Vec<_>>() })))
}

async fn query_graph(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<QueryBody>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let limit = body.limit.unwrap_or(10).min(semdoc::query::MAX_LIMIT);
    let want_answer = body.answer.unwrap_or(false);
    // Filter is validated (compile errors surface as 400) but graph queries
    // don't consume SQL — the graph backend has its own metadata filtering.
    let _ = filter_sql(&state, &body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let mode = if want_answer { GraphMode::Hybrid } else { GraphMode::Data };
    let params = graph_params(&body);
    let answer = match state.mcp.graph.query(&body.text, mode, limit, &params).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[graph] degraded to semantic: {e:#}");
            let docs = state
                .mcp.engine
                .query_semantic(&body.text, limit, None, ExpandTo::Chunk)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
            return Ok(Json(json!({
                "degraded": true,
                "answer": Value::Null,
                "graph": Value::Null,
                "documents": docs.iter().map(record_json).collect::<Vec<_>>(),
            })));
        }
    };

    if !want_answer {
        // Data mode: parse the structured envelope, map data.chunks[].doc_id
        // back to local parent documents (lightrag full_doc_id == our parent
        // id), preserving lightrag's relevance order. Filter applies to the
        // mapped documents.
        let raw: Value = serde_json::from_str(&answer.content)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("graph data decode: {e}")))?;
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
            let parents = state
                .mcp.engine
                .store
                .get_parents(&chunk_doc_ids)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
            for id in &chunk_doc_ids {
                if let Some(p) = parents.get(id) {
                    if let Some(fj) = &body.filter {
                        if let Ok(pred) = semdoc::query::Pred::from_json(fj) {
                            if !pred_matches(&pred, p) {
                                continue;
                            }
                        }
                    }
                    documents.push(record_json(p));
                }
            }
        }
        // Fallback mapping when the lightrag HTTP build doesn't attach
        // doc_id to chunks (only the semrag Python wrapper does): match
        // chunk content against parent raw_text. The doc-count is small
        // (hundreds) so an in-memory substring scan is fine.
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
                let all_parents = state
                    .mcp.engine
                    .store
                    .all_parents()
                    .await
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
                for p in &all_parents {
                    if documents.len() >= limit {
                        break;
                    }
                    if contents.iter().any(|c| p.raw_text.contains(c.as_str())) {
                        if let Some(fj) = &body.filter {
                            if let Ok(pred) = semdoc::query::Pred::from_json(fj) {
                                if !pred_matches(&pred, p) {
                                    continue;
                                }
                            }
                        }
                        documents.push(record_json(p));
                    }
                }
            }
        }
        documents.truncate(limit);
        return Ok(Json(json!({
            "degraded": false,
            "graph": data,
            "documents": documents,
        })));
    }

    Ok(Json(json!({
        "degraded": false,
        "answer": answer.content,
    })))
}

/// /query/hybrid — atomic (reranked semantic) + graph retrieval in parallel,
/// merged + deduped by id. Graph side first (synthesized summary / mapped
/// docs), atomic side as supporting evidence. Filter applies to the atomic
/// side; graph side runs unfiltered.
async fn query_hybrid(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<QueryBody>,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let limit = body.limit.unwrap_or(10).min(semdoc::query::MAX_LIMIT);
    let sql = filter_sql(&state, &body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let params = graph_params(&body);
    let want_answer = body.answer.unwrap_or(false);
    let mode = if want_answer { GraphMode::Hybrid } else { GraphMode::Data };

    let atomic_fut = state.mcp.engine.query_reranked(&body.text, limit, sql.as_deref());
    let graph_fut = state.mcp.graph.query(&body.text, mode, limit, &params);

    let (atomic_res, graph_res) = tokio::join!(atomic_fut, graph_fut);

    // Graph side: degrade to empty (semantic already covered by atomic side).
    let mut merged: Vec<Value> = Vec::new();
    let mut degraded = false;
    let mut answer_json = Value::Null;
    let mut graph_json = Value::Null;
    match graph_res {
        Ok(a) => {
            if want_answer {
                answer_json = Value::String(a.content);
            } else if let Ok(raw) = serde_json::from_str::<Value>(&a.content) {
                graph_json = raw.get("data").cloned().unwrap_or(Value::Null);
                // Map docs but don't dedupe against atomic here — merge below
                // by id handles it.
            }
        }
        Err(e) => {
            eprintln!("[hybrid] graph side degraded: {e:#}");
            degraded = true;
        }
    }

    let mut seen = std::collections::HashSet::new();
    // Graph docs first when in data mode (relevance-ordered), then atomic.
    if !want_answer {
        if let Ok(raw) = &graph_json.clone().to_string().parse::<Value>() {
            let _ = raw; // graph docs mapping done below via graph_json
        }
        if let Some(data) = graph_json.as_object() {
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
                if let Ok(parents) = state.mcp.engine.store.get_parents(&chunk_doc_ids).await {
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
                // Content-matching fallback (lightrag HTTP builds without doc_id).
                if let Ok(all) = state.mcp.engine.store.all_parents().await {
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
    } else {
        merged.push(json!({ "answer": answer_json }));
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
    Ok(Json(json!({
        "degraded": degraded,
        "answer": answer_json,
        "graph": graph_json,
        "documents": merged,
    })))
}

/// Liveness/readiness probe. No auth (probes don't carry tokens); no graph
/// health call (must answer in milliseconds even when lightrag is down).
async fn health(State(state): State<Arc<AppState>>) -> Resp<Value> {
    let rows = state.mcp.engine.store.count().await.unwrap_or(0);
    Ok(Json(json!({
        "ok": true,
        "rows": rows,
        "graph": state.mcp.graph.name(),
    })))
}

async fn stats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let rows = state
        .mcp.engine
        .store
        .count()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(Json(json!({"rows": rows})))
}

fn graph_params(body: &QueryBody) -> semdoc::plugins::graph::GraphQueryParams {
    semdoc::plugins::graph::GraphQueryParams {
        chunk_top_k: body.chunk_top_k,
        max_entity_tokens: body.max_entity_tokens,
        max_relation_tokens: body.max_relation_tokens,
        max_total_tokens: body.max_total_tokens,
    }
}

/// In-memory predicate evaluation for the graph-data document mapping path
/// (the SQL path can't apply — mapping happens after the ANN/FTS engine).
fn pred_matches(pred: &semdoc::query::Pred, rec: &semdoc::store::Record) -> bool {
    use semdoc::query::Pred;
    match pred {
        Pred::And(ps) => ps.iter().all(|p| pred_matches(p, rec)),
        Pred::Or(ps) => ps.iter().any(|p| pred_matches(p, rec)),
        Pred::Exists(name, yes) => {
            let has = RESERVED_FILTERABLE_MEME.contains(&name.as_str())
                || rec.extra.contains_key(name);
            has == *yes
        }
        _ => {
            // Scalar comparisons via JSON: fetch the field value and compare.
            let (name, want, op) = match pred {
                Pred::Eq(n, v) => (n, v, 0),
                Pred::Ne(n, v) => (n, v, 1),
                Pred::Gt(n, v) => (n, v, 2),
                Pred::Gte(n, v) => (n, v, 3),
                Pred::Lt(n, v) => (n, v, 4),
                Pred::Lte(n, v) => (n, v, 5),
                Pred::In(n, vs) => {
                    return vs.iter().any(|v| json_eq(&field_value(rec, n), v));
                }
                Pred::Nin(n, vs) => {
                    return !vs.iter().any(|v| json_eq(&field_value(rec, n), v));
                }
                _ => return true, // Raw: SQL-only, allow through
            };
            let got = field_value(rec, name);
            let eq = json_eq(&got, want);
            match op {
                0 => eq,
                1 => !eq,
                _ => match (got.as_f64(), want.as_f64()) {
                    (Some(a), Some(b)) => match op {
                        2 => a > b,
                        3 => a >= b,
                        4 => a < b,
                        5 => a <= b,
                        _ => false,
                    },
                    _ => false,
                },
            }
        }
    }
}

const RESERVED_FILTERABLE_MEME: &[&str] = &["source_path", "source_type"];

fn field_value(rec: &semdoc::store::Record, name: &str) -> Value {
    match name {
        "source_path" => Value::String(rec.source_path.clone()),
        "source_type" => Value::String(rec.source_type.clone()),
        _ => rec.extra.get(name).cloned().unwrap_or(Value::Null),
    }
}

fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            x.as_f64() == y.as_f64() // f32 storage vs int compare tolerance
        }
        _ => a == b,
    }
}

/// Shared write path helpers (kept out of the lib to avoid duplicating the
/// CLI's copy — see semdoc.rs `write_doc`; both are thin wrappers around
/// the same store calls).
mod semdoc_bin_helpers {
    use anyhow::Result;
    use semdoc::embedding::Embedder;
    use semdoc::store::{InputDoc, Record, Store};
    use std::collections::HashMap;

    pub async fn write_doc(
        store: &Store,
        embedder: &Embedder,
        doc: InputDoc,
    ) -> Result<()> {
        let doc_id = doc.id.clone();
        let language = doc.language.clone();
        let source_path = doc.source_path.clone();
        let leaves_src =
            semdoc::chunker::chunk_for(&doc.raw_text, &doc_id, language.as_deref(), &source_path);

        let mut parent_vectors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut leaf_vector_lists: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

        for vf in &store.vector_fields {
            if !vf.auto_embed {
                continue;
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
            let parent_vec = embedder.encode_blocking(src_text).await?;
            if parent_vec.len() != vf.dim {
                anyhow::bail!(
                    "vector `{}`: embedder dim {} != configured {}",
                    vf.name,
                    parent_vec.len(),
                    vf.dim
                );
            }
            let mut lvs = Vec::with_capacity(leaves_src.len());
            for c in &leaves_src {
                let v = embedder.encode_blocking(c.text.clone()).await?;
                lvs.push(v);
            }
            parent_vectors.insert(vf.name.clone(), parent_vec);
            leaf_vector_lists.insert(vf.name.clone(), lvs);
        }

        let parent = Record {
            id: doc_id.clone(),
            raw_text: doc.raw_text.clone(),
            chunk_level: 0,
            chunk_index: 0,
            parent_doc_id: None,
            source_path: source_path.clone(),
            source_type: doc.source_type.clone(),
            extra: doc.extra.clone(),
        };
        let leaves: Vec<Record> = leaves_src
            .iter()
            .map(|c| Record {
                id: c.chunk_id.clone(),
                raw_text: c.text.clone(),
                chunk_level: 1,
                chunk_index: c.chunk_index,
                parent_doc_id: Some(doc_id.clone()),
                source_path: source_path.clone(),
                source_type: doc.source_type.clone(),
                extra: doc.extra.clone(),
            })
            .collect();

        let mut rows: Vec<(Record, HashMap<String, Vec<f32>>)> =
            vec![(parent, parent_vectors.clone())];
        for (i, l) in leaves.iter().enumerate() {
            let mut lv: HashMap<String, Vec<f32>> = HashMap::new();
            for vf in &store.vector_fields {
                if let Some(list) = leaf_vector_lists.get(&vf.name) {
                    if let Some(v) = list.get(i) {
                        lv.insert(vf.name.clone(), v.clone());
                    }
                }
            }
            rows.push((l.clone(), lv));
        }
        let refs: Vec<(&Record, &HashMap<String, Vec<f32>>)> =
            rows.iter().map(|(r, v)| (r, v)).collect();
        let batch = store.build_batch_public(&refs)?;
        store.write_batch(batch).await?;
        Ok(())
    }
}

// ── MCP streamable HTTP transport ─────────────────────────────────────

const MCP_SESSION_TTL: Duration = Duration::from_secs(24 * 60 * 60);
const MCP_SWEEP_PERIOD: Duration = Duration::from_secs(15 * 60);

use std::time::Duration;

/// Per-session state for the MCP streamable HTTP transport. Created by
/// `initialize`, looked up by `Mcp-Session-Id` on subsequent requests,
/// removed by DELETE /mcp or idle-TTL expiry.
#[derive(Clone)]
struct McpSession {
    initialized: bool,
    protocol_version: String,
    created_at: Instant,
    last_seen: Instant,
}

/// In-memory session map mirrored to SQLite (`mcp_sessions` table) so a
/// server restart rehydrates sessions instead of kicking every client.
/// `last_seen` slides forward on every successful `get` — active sessions
/// never expire; TTL is an idle timeout, not a lifetime cap.
struct McpSessionStore {
    sessions: tokio::sync::RwLock<std::collections::HashMap<String, McpSession>>,
    db_path: String,
}

impl McpSessionStore {
    fn open(db_path: &str) -> Self {
        if let Ok(conn) = rusqlite::Connection::open(db_path) {
            let _ = conn.execute(
                "CREATE TABLE IF NOT EXISTS mcp_sessions (
                    id               TEXT PRIMARY KEY,
                    initialized      INTEGER NOT NULL,
                    protocol_version TEXT NOT NULL,
                    created_at       INTEGER NOT NULL,
                    last_seen        INTEGER NOT NULL
                )",
                [],
            );
        }
        let mut map = std::collections::HashMap::new();
        if let Ok(conn) = rusqlite::Connection::open(db_path) {
            if let Ok(mut stmt) = conn.prepare("SELECT id, initialized, protocol_version FROM mcp_sessions") {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                }) {
                    let now = Instant::now();
                    for r in rows.flatten() {
                        map.insert(
                            r.0,
                            McpSession {
                                initialized: r.1 != 0,
                                protocol_version: r.2,
                                created_at: now,
                                last_seen: now,
                            },
                        );
                    }
                }
            }
        }
        let n = map.len();
        if n > 0 {
            eprintln!("[mcp] rehydrated {n} session(s) from {db_path}");
        }
        Self {
            sessions: tokio::sync::RwLock::new(map),
            db_path: db_path.to_string(),
        }
    }

    async fn get(&self, id: &str) -> Option<McpSession> {
        let now = Instant::now();
        let mut map = self.sessions.write().await;
        if let Some(s) = map.get_mut(id) {
            s.last_seen = now;
            return Some(s.clone());
        }
        None
    }

    async fn insert(&self, id: String, session: McpSession) {
        self.sessions.write().await.insert(id.clone(), session.clone());
        let path = self.db_path.clone();
        tokio::task::spawn_blocking(move || {
            if let Ok(conn) = rusqlite::Connection::open(&path) {
                let _ = conn.execute(
                    "INSERT OR REPLACE INTO mcp_sessions (id, initialized, protocol_version, created_at, last_seen) VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![
                        id,
                        session.initialized as i64,
                        session.protocol_version,
                        session.created_at.elapsed().as_secs() as i64,
                        session.last_seen.elapsed().as_secs() as i64
                    ],
                );
            }
        });
    }

    async fn remove(&self, id: &str) -> bool {
        let existed = self.sessions.write().await.remove(id).is_some();
        if existed {
            let path = self.db_path.clone();
            let id = id.to_string();
            tokio::task::spawn_blocking(move || {
                if let Ok(conn) = rusqlite::Connection::open(&path) {
                    let _ = conn.execute("DELETE FROM mcp_sessions WHERE id = ?1", [id]);
                }
            });
        }
        existed
    }

    async fn ttl_sweep(self: Arc<Self>, ttl: Duration, period: Duration) {
        let mut interval = tokio::time::interval(period);
        loop {
            interval.tick().await;
            let now = Instant::now();
            let mut map = self.sessions.write().await;
            let mut stale_ids: Vec<String> = Vec::new();
            map.retain(|id, s| {
                if now.duration_since(s.last_seen) < ttl {
                    true
                } else {
                    stale_ids.push(id.clone());
                    false
                }
            });
            let evicted = stale_ids.len();
            drop(map);
            if evicted > 0 {
                eprintln!("[mcp] TTL sweep evicted {evicted} stale session(s)");
                let path = self.db_path.clone();
                tokio::task::spawn_blocking(move || {
                    if let Ok(conn) = rusqlite::Connection::open(&path) {
                        for id in &stale_ids {
                            let _ = conn.execute("DELETE FROM mcp_sessions WHERE id = ?1", [id]);
                        }
                    }
                });
            }
        }
    }
}

fn token_matches(received: &str, expected: &str) -> bool {
    use std::hint::black_box;
    if received.len() != expected.len() {
        let _ = black_box(received.bytes().zip(expected.bytes()).map(|(a, b)| a ^ b));
        return false;
    }
    let mut acc: u8 = 0;
    for (a, b) in received.bytes().zip(expected.bytes()) {
        acc |= a ^ b;
    }
    black_box(acc) == 0
}

fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    let hv = headers.get(axum::http::header::AUTHORIZATION)?;
    let s = hv.to_str().ok()?;
    s.strip_prefix("Bearer ").map(|t| t.trim())
}

fn envelope(id: &Value, result: Value) -> Value {
    if let Some(err) = result.get("error") {
        json!({ "jsonrpc": "2.0", "id": id, "error": err })
    } else {
        json!({ "jsonrpc": "2.0", "id": id, "result": result })
    }
}

fn uuid_str() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:016x}{:016x}", now.as_nanos(), n)
}

/// POST /mcp — streamable HTTP entry. JSON-RPC in; JSON or SSE out.
async fn mcp_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let expected = match &state.expected_token {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                "SEMDOC_MCP_TOKEN not set; /mcp disabled",
            )
                .into_response();
        }
    };
    match extract_bearer(&headers) {
        Some(received) if token_matches(received, expected) => {}
        _ => return (StatusCode::UNAUTHORIZED, "Invalid or missing bearer token").into_response(),
    }

    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let body = serde_json::to_string(&json!({
                "jsonrpc": "2.0", "id": null,
                "error": { "code": -32700, "message": "Parse error", "data": e.to_string() }
            }))
            .unwrap();
            return (
                StatusCode::BAD_REQUEST,
                [("content-type", "application/json")],
                body,
            )
                .into_response();
        }
    };

    let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let params = req.get("params").cloned().unwrap_or(json!({}));
    let id = req.get("id").cloned().unwrap_or(json!(null));

    // Session enforcement: initialize mints a session; everything else
    // requires a valid Mcp-Session-Id (400, not 404 — endpoint exists).
    if method != "initialize" {
        let sid = headers
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if state.mcp_sessions.get(sid).await.is_none() {
            return (
                StatusCode::BAD_REQUEST,
                "Invalid or missing Mcp-Session-Id. Call initialize first.",
            )
                .into_response();
        }
    }

    let result = match method {
        "initialize" | "notifications/initialized" | "tools/list" | "tools/call" => {
            semdoc::mcp::handle_request(&state.mcp, method, &params).await
        }
        other => json!({
            "error": { "code": -32601, "message": format!("method not found: {other}") }
        }),
    };
    let response_value = envelope(&id, result);

    // initialize: mint a session, return it via Mcp-Session-Id header.
    let session_header = if method == "initialize" {
        let sid = uuid_str();
        let now = Instant::now();
        state
            .mcp_sessions
            .insert(
                sid.clone(),
                McpSession {
                    initialized: true,
                    protocol_version: params
                        .get("protocolVersion")
                        .and_then(|v| v.as_str())
                        .unwrap_or("2024-11-05")
                        .to_string(),
                    created_at: now,
                    last_seen: now,
                },
            )
            .await;
        Some(sid)
    } else {
        None
    };

    // Respond JSON when the client accepts it; otherwise wrap in one SSE
    // `message` event (single response, no progress notifications).
    let prefer_json = headers
        .get(axum::http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .map(|s| {
            s.split(',').any(|t| {
                let t = t.trim();
                t.eq_ignore_ascii_case("application/json") || t.starts_with("application/json;")
            })
        })
        .unwrap_or(true);

    let body_str = serde_json::to_string(&response_value).unwrap();
    let mut resp = if prefer_json {
        (
            StatusCode::OK,
            [("content-type", "application/json")],
            body_str,
        )
            .into_response()
    } else {
        let sse = format!("event: message\ndata: {body_str}\n\n");
        (
            StatusCode::OK,
            [
                ("content-type", "text/event-stream"),
                ("cache-control", "no-cache"),
            ],
            sse,
        )
            .into_response()
    };
    if let Some(sid) = session_header {
        if let Ok(hv) = axum::http::HeaderValue::from_str(&sid) {
            resp.headers_mut().insert("mcp-session-id", hv);
        }
    }
    resp
}

/// DELETE /mcp — terminate a session (spec: client sends DELETE with
/// Mcp-Session-Id; server removes it and returns 200).
async fn mcp_delete(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let expected = match &state.expected_token {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                "SEMDOC_MCP_TOKEN not set; /mcp disabled",
            )
                .into_response();
        }
    };
    match extract_bearer(&headers) {
        Some(received) if token_matches(received, expected) => {}
        _ => return (StatusCode::UNAUTHORIZED, "Invalid or missing bearer token").into_response(),
    }
    let sid = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    state.mcp_sessions.remove(sid).await;
    StatusCode::OK.into_response()
}
