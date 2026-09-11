// SPDX-License-Identifier: MIT OR Apache-2.0
//! semdoc-server — HTTP REST + graph plugin endpoints.
//!
//!   POST /documents            add doc (auto-embed)
//!   POST /query/semantic       {text, limit, filter?, expand_to?}
//!   POST /query/text           {text, limit, filter?}
//!   POST /query/reranked       {text, limit, filter?}
//!   POST /query/graph          {text, limit} — via graph plugin, degrade to semantic
//!   POST /documents/delete     {id}
//!   GET  /stats
//!
//! `filter` is the Mongo-style JSON DSL compiled against the schema.

use anyhow::Result;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use semdoc::plugins::graph::{build_graph_plugin, GraphMode, GraphPlugin};
use semdoc::query::{record_json, Engine, ExpandTo};
use semdoc::schema::SchemaConfig;
use semdoc::store::{InputDoc, Store};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

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
    engine: Engine,
    graph: Box<dyn GraphPlugin>,
    token: Option<String>,
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

fn auth_err() -> impl IntoResponse {
    (StatusCode::UNAUTHORIZED, "unauthorized")
}

fn filter_sql(state: &AppState, body: &QueryBody) -> Result<Option<String>, String> {
    match &body.filter {
        Some(f) if f.as_object().is_some_and(|o| !o.is_empty()) => {
            let pred = semdoc::query::Pred::from_json(f).map_err(|e| e.to_string())?;
            pred.compile(&state.engine.config).map(Some).map_err(|e| e.to_string())
        }
        _ => Ok(None),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg_path = format!("{}/schema.toml", args.db);
    let config = SchemaConfig::load(std::path::Path::new(&cfg_path))?;
    let vec_fields = config.effective_vector_fields(1024);
    let conn = lancedb::connect(&args.db).execute().await?;
    let store = Store::new_existing(
        conn,
        config.table.name.clone(),
        Store::arrow_schema(&config, &vec_fields),
        vec_fields.clone(),
        config.fields.clone(),
    );
    let deploy = semdoc::config::DeploymentConfig::load(args.config.as_deref())?;
    deploy.apply_chunk_env();
    let embedder = semdoc::embedding::Embedder::load(&deploy.embedding)?;
    let reranker = if args.rerank {
        Some(Arc::new(semdoc::reranker::Reranker::load(&deploy.rerank)?))
    } else {
        None
    };
    let graph = build_graph_plugin(&config.plugins.graph).await?;
    let token = deploy
        .server_token()
        .or_else(|| std::env::var("SEMDOC_TOKEN").ok())
        .or(if args.auth {
            None // --auth set but no token found → error below
        } else {
            None
        });
    if args.auth && token.is_none() {
        anyhow::bail!("--auth requires server.token_env (config) or SEMDOC_TOKEN (env)");
    }
    let state = Arc::new(AppState {
        engine: Engine { store, embedder, reranker, config },
        graph,
        token,
    });

    let app = Router::new()
        .route("/documents", post(add_doc))
        .route("/documents/delete", post(delete_doc))
        .route("/query/semantic", post(query_semantic))
        .route("/query/text", post(query_text))
        .route("/query/reranked", post(query_reranked))
        .route("/query/graph", post(query_graph))
        .route("/query/hybrid", post(query_hybrid))
        .route("/stats", get(stats))
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
    let doc = InputDoc {
        id: blake3::hash(body.text.as_bytes()).to_hex().to_string(),
        raw_text: body.text.clone(),
        source_path: body.source_path.unwrap_or_else(|| "inline".into()),
        source_type: "text".into(),
        language: Some("markdown".into()),
        extra: body.meta,
        vectors: Default::default(),
    };
    semdoc_bin_helpers::write_doc(&state.engine.store, &state.engine.config, &state.engine.embedder, doc)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
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
        .engine
        .store
        .delete_by_id(&body.id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(Json(json!({"ok": true})))
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
        .engine
        .query_semantic(&body.text, body.limit.unwrap_or(10), sql.as_deref(), expand_to)
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
        .engine
        .query_fts(&body.text, body.limit.unwrap_or(10), sql.as_deref())
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
        .engine
        .query_reranked(&body.text, body.limit.unwrap_or(10), sql.as_deref())
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
    let limit = body.limit.unwrap_or(10);
    let want_answer = body.answer.unwrap_or(false);
    let sql = filter_sql(&state, &body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let mode = if want_answer { GraphMode::Hybrid } else { GraphMode::Data };
    let params = graph_params(&body);
    let answer = match state.graph.query(&body.text, mode, limit, &params).await {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[graph] degraded to semantic: {e:#}");
            let docs = state
                .engine
                .query_semantic(&body.text, limit, None, ExpandTo::Chunk)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
            return Ok(Json(json!({
                "degraded": true,
                "answer": if want_answer { Value::Null } else { Value::Null },
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
                .engine
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
                    .engine
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
    let limit = body.limit.unwrap_or(10);
    let sql = filter_sql(&state, &body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let params = graph_params(&body);
    let want_answer = body.answer.unwrap_or(false);
    let mode = if want_answer { GraphMode::Hybrid } else { GraphMode::Data };

    let atomic_fut = state.engine.query_reranked(&body.text, limit, sql.as_deref());
    let graph_fut = state.graph.query(&body.text, mode, limit, &params);

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
                if let Ok(parents) = state.engine.store.get_parents(&chunk_doc_ids).await {
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
                if let Ok(all) = state.engine.store.all_parents().await {
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

async fn stats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Resp<Value> {
    if !auth_ok(&state, &headers) {
        return Err((StatusCode::UNAUTHORIZED, "unauthorized".into()));
    }
    let rows = state
        .engine
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
    use semdoc::schema::SchemaConfig;
    use semdoc::store::{InputDoc, Record, Store};
    use std::collections::HashMap;

    pub async fn write_doc(
        store: &Store,
        config: &SchemaConfig,
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
