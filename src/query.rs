// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filter DSL + compilation to LanceDB `only_if` SQL, and the query engine
//! (semantic small-to-big, FTS, reranked) over a schema-driven [`Store`].

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

use crate::reranker::Reranker;
use crate::schema::{FieldConfig, SchemaConfig};
use crate::store::{Record, Store};

/// Reserved scalar columns usable in filters without being in `[fields]`.
const RESERVED_FILTERABLE: &[&str] = &["source_path", "source_type"];

// ---------------------------------------------------------------------------
// Filter DSL — Mongo-style, evaluated against a schema
// ---------------------------------------------------------------------------

/// A parsed filter expression. Values are JSON; types are checked against
/// the schema at compile time (`compile`), so a string compared against an
/// int64 field is an error, not a silent no-match.
#[derive(Debug, Clone, PartialEq)]
pub enum Pred {
    Eq(String, Value),
    Ne(String, Value),
    In(String, Vec<Value>),
    Nin(String, Vec<Value>),
    Gt(String, Value),
    Gte(String, Value),
    Lt(String, Value),
    Lte(String, Value),
    Exists(String, bool),
    /// Raw SQL escape hatch — spliced verbatim into `only_if`. Trusted-input
    /// only (CLI flag / server admin), never exposed over MCP.
    Raw(String),
    And(Vec<Pred>),
    Or(Vec<Pred>),
}

impl Pred {
    /// Parse from a Mongo-style JSON object:
    /// `{"category": "a", "score": {"$gte": 3}, "$or": [{"a": 1}, {"b": 2}]}`
    pub fn from_json(v: &Value) -> Result<Pred> {
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow!("filter must be a JSON object"))?;
        let mut preds = Vec::new();
        for (k, val) in obj {
            match k.as_str() {
                "$and" => {
                    let arr = val.as_array().ok_or_else(|| anyhow!("$and must be an array"))?;
                    preds.push(Pred::And(
                        arr.iter().map(Pred::from_json).collect::<Result<Vec<_>>>()?,
                    ));
                }
                "$or" => {
                    let arr = val.as_array().ok_or_else(|| anyhow!("$or must be an array"))?;
                    preds.push(Pred::Or(
                        arr.iter().map(Pred::from_json).collect::<Result<Vec<_>>>()?,
                    ));
                }
                _ => match val {
                    Value::Object(ops) if ops.keys().all(|k| k.starts_with('$')) => {
                        for (op, opval) in ops {
                            let p = match op.as_str() {
                                "$eq" => Pred::Eq(k.clone(), opval.clone()),
                                "$ne" => Pred::Ne(k.clone(), opval.clone()),
                                "$in" => Pred::In(
                                    k.clone(),
                                    opval.as_array().ok_or_else(|| anyhow!("$in needs array"))?
                                        .clone(),
                                ),
                                "$nin" => Pred::Nin(
                                    k.clone(),
                                    opval.as_array().ok_or_else(|| anyhow!("$nin needs array"))?
                                        .clone(),
                                ),
                                "$gt" => Pred::Gt(k.clone(), opval.clone()),
                                "$gte" => Pred::Gte(k.clone(), opval.clone()),
                                "$lt" => Pred::Lt(k.clone(), opval.clone()),
                                "$lte" => Pred::Lte(k.clone(), opval.clone()),
                                "$exists" => Pred::Exists(k.clone(), opval.as_bool().unwrap_or(true)),
                                other => anyhow::bail!("unknown operator {other} for field {k}"),
                            };
                            preds.push(p);
                        }
                    }
                    other => preds.push(Pred::Eq(k.clone(), other.clone())),
                },
            }
        }
        if preds.len() == 1 {
            Ok(preds.pop().unwrap())
        } else {
            Ok(Pred::And(preds))
        }
    }

    /// Validate field names + value types against the schema and compile to
    /// `only_if` SQL.
    pub fn compile(&self, config: &SchemaConfig) -> Result<String> {
        let mut parts = Vec::new();
        self.compile_into(config, &mut parts)?;
        if parts.is_empty() {
            Ok("TRUE".to_string())
        } else {
            Ok(parts.join(" AND "))
        }
    }

    fn compile_into(&self, config: &SchemaConfig, out: &mut Vec<String>) -> Result<()> {
        match self {
            Pred::And(ps) => {
                for p in ps {
                    p.compile_into(config, out)?;
                }
            }
            Pred::Or(ps) => {
                let mut branches = Vec::new();
                for p in ps {
                    let mut sub = Vec::new();
                    p.compile_into(config, &mut sub)?;
                    branches.push(format!("({})", sub.join(" AND ")));
                }
                out.push(format!("({})", branches.join(" OR ")));
            }
            Pred::Raw(sql) => out.push(sql.clone()),
            _ => out.push(self.compile_atom(config)?),
        }
        Ok(())
    }

    fn field_type<'a>(&self, config: &'a SchemaConfig, name: &str) -> Result<&'a str> {
        if let Some(fc) = config.fields.get(name) {
            return Ok(&fc.r#type);
        }
        if RESERVED_FILTERABLE.contains(&name) {
            return Ok("string");
        }
        if ["chunk_level", "chunk_index"].contains(&name) {
            return Ok("int64");
        }
        Err(anyhow!("unknown filter field `{name}` — not in [fields] and not reserved"))
    }

    fn compile_atom(&self, config: &SchemaConfig) -> Result<String> {
        let (name, val, op) = match self {
            Pred::Eq(n, v) => (n, v, "="),
            Pred::Ne(n, v) => (n, v, "!="),
            Pred::Gt(n, v) => (n, v, ">"),
            Pred::Gte(n, v) => (n, v, ">="),
            Pred::Lt(n, v) => (n, v, "<"),
            Pred::Lte(n, v) => (n, v, "<="),
            Pred::In(n, vs) => return compile_in(config, n, vs, true),
            Pred::Nin(n, vs) => return compile_in(config, n, vs, false),
            Pred::Exists(n, yes) => {
                self.field_type(config, n)?; // validate name
                return Ok(if *yes {
                    format!("{n} IS NOT NULL")
                } else {
                    format!("{n} IS NULL")
                });
            }
            _ => unreachable!("compile_atom on compound pred"),
        };
        let t = self.field_type(config, name)?;
        let lit = value_to_sql(t, val)?;
        Ok(format!("{name} {op} {lit}"))
    }
}

fn compile_in(config: &SchemaConfig, name: &str, vals: &[Value], positive: bool) -> Result<String> {
    let t = match config.fields.get(name) {
        Some(fc) => fc.r#type.as_str(),
        None if RESERVED_FILTERABLE.contains(&name) => "string",
        None => return Err(anyhow!("unknown filter field `{name}`")),
    };
    if t == "list<string>" {
        // array_has(name, v1) OR array_has(name, v2) ...
        let conds: Vec<String> = vals
            .iter()
            .map(|v| {
                let s = v
                    .as_str()
                    .ok_or_else(|| anyhow!("list<string> membership needs string values"))?;
                Ok(format!("array_has({name}, '{}')", s.replace('\'', "''")))
            })
            .collect::<Result<Vec<_>>>()?;
        let joined = format!("({})", conds.join(" OR "));
        return Ok(if positive { joined } else { format!("NOT {joined}") });
    }
    let lits: Vec<String> = vals.iter().map(|v| value_to_sql(t, v)).collect::<Result<_>>()?;
    let kw = if positive { "IN" } else { "NOT IN" };
    Ok(format!("{name} {kw} ({})", lits.join(", ")))
}

fn value_to_sql(t: &str, v: &Value) -> Result<String> {
    let _ = t;
    value_to_sql_checked(t, v)
}

fn value_to_sql_checked(t: &str, v: &Value) -> Result<String> {
    match t {
        "string" | "text" => {
            let s = v.as_str().ok_or_else(|| anyhow!("field is {t}, value must be a string"))?;
            Ok(format!("'{}'", s.replace('\'', "''")))
        }
        "int64" | "timestamp" => {
            let n = v.as_i64().ok_or_else(|| anyhow!("field is {t}, value must be an integer"))?;
            Ok(n.to_string())
        }
        "float32" => {
            let n = v.as_f64().ok_or_else(|| anyhow!("field is float32, value must be a number"))?;
            Ok(format!("{n}"))
        }
        "bool" => {
            let b = v.as_bool().ok_or_else(|| anyhow!("field is bool, value must be boolean"))?;
            Ok(if b { "TRUE" } else { "FALSE" }.to_string())
        }
        "list<string>" => {
            // List membership uses array_has(name, 'value') — needs the
            // column name, handled by compile_in; bare literals unsupported.
            Err(anyhow!(
                "list<string> only supports $in/$nin membership filters"
            ))
        }
        other => Err(anyhow!("cannot filter on type {other}")),
    }
}

// ---------------------------------------------------------------------------
// Query engine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpandTo {
    Chunk,
    #[default]
    Parent,
    Auto,
}

impl ExpandTo {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "chunk" => Self::Chunk,
            "auto" => Self::Auto,
            _ => Self::Parent,
        }
    }
}

/// Cap raw_text in API responses (MCP/HTTP). Prevents parent-expanded hits
/// from blowing client token budgets.
pub const MAX_RAW_TEXT_CHARS: usize = 2000;

pub fn preview(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let mut s: String = text.chars().take(max_chars).collect();
        s.push_str("...");
        s
    }
}

/// Over-recall pool size for rerank (leaf chunks now — cheap).
pub const RERANK_RECALL_K: usize = 80;

/// Cap on `limit` for API requests (MCP/HTTP). Prevents a single request
/// with limit=100000 from triggering a full-table ANN + massive response.
/// CLI is uncapped (trusted local caller).
pub const MAX_LIMIT: usize = 200;

pub struct Engine {
    pub store: Store,
    pub embedder: crate::embedding::Embedder,
    pub reranker: Option<std::sync::Arc<Reranker>>,
    pub config: SchemaConfig,
}

impl Engine {
    pub fn primary_vector(&self) -> &crate::schema::VectorFieldConfig {
        &self.store.vector_fields[0]
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embedder.encode_blocking(text.to_string()).await?;
        let dim = self.primary_vector().dim;
        if v.len() != dim {
            anyhow::bail!(
                "embedder returned dim {} but vector column `{}` expects dim {}",
                v.len(),
                self.primary_vector().name,
                dim
            );
        }
        Ok(v)
    }

    /// Semantic query with small-to-big expansion.
    pub async fn query_semantic(
        &self,
        text: &str,
        limit: usize,
        filter_sql: Option<&str>,
        expand_to: ExpandTo,
    ) -> Result<Vec<Record>> {
        let v = self.embed(text).await?;
        let leaves = self
            .store
            .query_leaves(&self.primary_vector().name, &v, limit, filter_sql)
            .await?;
        expand_leaf_hits(&self.store, leaves, expand_to).await
    }

    /// Plain ANN (no expansion) — used by the filter fallback path.
    pub async fn query_ann(
        &self,
        text: &str,
        limit: usize,
        filter_sql: Option<&str>,
    ) -> Result<Vec<Record>> {
        let v = self.embed(text).await?;
        self.store
            .query_all(&self.primary_vector().name, &v, limit, filter_sql)
            .await
    }

    pub async fn query_fts(&self, text: &str, limit: usize, filter_sql: Option<&str>) -> Result<Vec<Record>> {
        self.store.query_fts(text, limit, filter_sql).await
    }

    /// Reranked query: leaf over-recall → cross-encoder → chunk results.
    /// Degrades to ANN order on rerank failure.
    pub async fn query_reranked(
        &self,
        text: &str,
        limit: usize,
        filter_sql: Option<&str>,
    ) -> Result<Vec<Record>> {
        let Some(reranker) = &self.reranker else {
            return self.query_semantic(text, limit, filter_sql, ExpandTo::Chunk).await;
        };
        let recall_k = limit.max(RERANK_RECALL_K);
        let v = self.embed(text).await?;
        let leaves = self
            .store
            .query_leaves(&self.primary_vector().name, &v, recall_k, filter_sql)
            .await?;
        if leaves.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = leaves.iter().map(|d| d.raw_text.clone()).collect();
        let q = text.to_string();
        let rk = reranker.clone();
        let n = leaves.len();
        let ranked = match tokio::task::spawn_blocking(move || rk.rerank(&q, &texts)).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                eprintln!("[rerank] failed, degrading to ANN order: {e:#}");
                (0..n).map(|i| (i, 0.0_f32)).collect()
            }
            Err(e) => {
                eprintln!("[rerank] join failed, degrading to ANN order: {e}");
                (0..n).map(|i| (i, 0.0_f32)).collect()
            }
        };
        let picked: Vec<Record> = ranked
            .into_iter()
            .take(limit)
            .filter_map(|(i, _)| leaves.get(i).cloned())
            .collect();
        Ok(picked)
    }
}

/// Expand leaf hits per small-to-big semantics (chunk / parent / auto).
pub async fn expand_leaf_hits(
    store: &Store,
    leaf_hits: Vec<Record>,
    expand_to: ExpandTo,
) -> Result<Vec<Record>> {
    if leaf_hits.is_empty() {
        return Ok(Vec::new());
    }
    let parent_ids: Vec<String> = leaf_hits.iter().filter_map(|h| h.parent_doc_id.clone()).collect();
    let parents = store.get_parents(&parent_ids).await?;

    use std::collections::{HashMap, HashSet};
    let mut by_parent: HashMap<String, Vec<&Record>> = HashMap::new();
    for h in &leaf_hits {
        if let Some(pid) = &h.parent_doc_id {
            by_parent.entry(pid.clone()).or_default().push(h);
        }
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<Record> = Vec::with_capacity(leaf_hits.len());
    for h in &leaf_hits {
        let Some(pid) = &h.parent_doc_id else {
            out.push(h.clone());
            continue;
        };
        if seen.contains(pid) {
            continue;
        }
        match expand_to {
            ExpandTo::Parent => match parents.get(pid) {
                Some(p) => {
                    out.push(p.clone());
                    seen.insert(pid.clone());
                }
                None => out.push(h.clone()),
            },
            ExpandTo::Chunk => out.push(h.clone()),
            ExpandTo::Auto => {
                let siblings = by_parent.get(pid).map(|v| v.len()).unwrap_or(0);
                if siblings >= 2 {
                    if let Some(p) = parents.get(pid) {
                        out.push(p.clone());
                        seen.insert(pid.clone());
                        continue;
                    }
                }
                if out.iter().any(|r| r.id == h.id) {
                    continue;
                }
                // Single hit: sentence-window — the hit merged with its ±1
                // sibling chunks into one synthesized record (parent_doc_id
                // preserved; id = first chunk's id so dedup-by-id works).
                let window = 1u32;
                let sibs = store
                    .get_sibling_chunks(pid, h.chunk_index, window)
                    .await
                    .unwrap_or_default();
                if sibs.is_empty() {
                    out.push(h.clone());
                    seen.insert(pid.clone());
                    continue;
                }
                let mut merged_text = String::new();
                let mut first_id = String::new();
                let mut first_idx = u32::MAX;
                let mut ordered: Vec<&Record> = sibs.iter().collect();
                ordered.push(h);
                ordered.sort_by_key(|r| r.chunk_index);
                for s in ordered {
                    if !merged_text.is_empty() {
                        merged_text.push_str("\n\n");
                    }
                    merged_text.push_str(&s.raw_text);
                    if s.chunk_index < first_idx {
                        first_idx = s.chunk_index;
                        first_id = s.id.clone();
                    }
                }
                out.push(Record {
                    id: first_id,
                    raw_text: merged_text,
                    chunk_level: h.chunk_level,
                    chunk_index: first_idx,
                    parent_doc_id: h.parent_doc_id.clone(),
                    source_path: h.source_path.clone(),
                    source_type: h.source_type.clone(),
                    extra: h.extra.clone(),
                });
                seen.insert(pid.clone());
            }
        }
    }
    Ok(out)
}

/// Record → JSON shape used by CLI/HTTP/MCP output.
pub fn record_json(d: &Record) -> Value {
    let (raw_text, truncated) = if d.raw_text.chars().count() > MAX_RAW_TEXT_CHARS {
        (preview(&d.raw_text, MAX_RAW_TEXT_CHARS), true)
    } else {
        (d.raw_text.clone(), false)
    };
    let mut obj = Map::new();
    obj.insert("id".into(), Value::String(d.id.clone()));
    obj.insert("raw_text".into(), Value::String(raw_text));
    obj.insert("truncated".into(), Value::Bool(truncated));
    obj.insert("chunk_level".into(), Value::Number(d.chunk_level.into()));
    obj.insert("chunk_index".into(), Value::Number(d.chunk_index.into()));
    obj.insert(
        "parent_doc_id".into(),
        d.parent_doc_id.clone().map(Value::String).unwrap_or(Value::Null),
    );
    obj.insert("source_path".into(), Value::String(d.source_path.clone()));
    obj.insert("source_type".into(), Value::String(d.source_type.clone()));
    for (k, v) in &d.extra {
        obj.insert(k.clone(), v.clone());
    }
    Value::Object(obj)
}
