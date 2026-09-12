// SPDX-License-Identifier: MIT OR Apache-2.0
//! semdoc CLI — schema-driven vector knowledge base.
//!
//!   semdoc init   --schema schema.toml --db ./mydb
//!   semdoc add    --db ./mydb --file doc.md [--meta category=a]
//!   semdoc query  --db ./mydb --text "..." [--semantic|--fts|--rerank] [--filter '{"category":"a"}']

use anyhow::Result;
use clap::{Parser, Subcommand};
use semdoc::query::{record_json, Engine, ExpandTo};
use semdoc::schema::SchemaConfig;
use semdoc::store::{InputDoc, Record, Store};

static DEPLOY: std::sync::OnceLock<semdoc::config::DeploymentConfig> = std::sync::OnceLock::new();

fn deploy() -> &'static semdoc::config::DeploymentConfig {
    DEPLOY.get_or_init(|| semdoc::config::DeploymentConfig::load(None).expect("load deployment config"))
}

#[derive(Parser)]
#[command(name = "semdoc", about = "Schema-driven vector knowledge base")]
struct Args {
    /// Deployment config file (default: $SEMDOC_CONFIG or ./semdoc.config.toml)
    #[arg(long, global = true)]
    config: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Initialize a database from a schema config
    Init {
        /// Path to schema.toml
        #[arg(short, long)]
        schema: String,
        /// Database directory
        #[arg(short, long)]
        db: String,
    },
    /// Add a document
    Add {
        #[arg(short, long)]
        db: String,
        /// File to add (markdown/plain text)
        #[arg(short, long)]
        file: Option<String>,
        /// Inline text
        #[arg(short = 't', long)]
        text: Option<String>,
        /// Source path label
        #[arg(short, long, default_value = "inline")]
        source: String,
        /// Extra field values, KEY=VALUE (repeatable)
        #[arg(long = "meta", value_delimiter = ',', action = clap::ArgAction::Append)]
        metas: Vec<String>,
    },
    /// Query
    Query {
        #[arg(short, long)]
        db: String,
        /// Query text
        #[arg(short = 't', long)]
        text: String,
        /// Semantic (default) | fts | rerank
        #[arg(long, default_value = "semantic")]
        mode: String,
        /// Max results
        #[arg(short, long, default_value = "5")]
        limit: usize,
        /// Filter JSON (Mongo-style: {"category":"a","score":{"$gte":3}})
        #[arg(long)]
        filter: Option<String>,
        /// Filter as raw SQL (trusted input only)
        #[arg(long)]
        filter_sql: Option<String>,
    },
    /// Show database stats
    Stats {
        #[arg(short, long)]
        db: String,
    },
}

async fn open_store(db: &str) -> Result<(semdoc::store::Store, SchemaConfig)> {
    let cfg_path = format!("{db}/schema.toml");
    let config = SchemaConfig::load(std::path::Path::new(&cfg_path))
        .map_err(|e| anyhow::anyhow!("load schema from {cfg_path}: {e}"))?;
    check_schema_compat(db, &config).await?;
    let vec_fields = config.effective_vector_fields(1024);
    let conn = lancedb::connect(db).execute().await?;
    // Reconstruct the Store without touching indexes — `init` owns creation.
    let store = semdoc::store::Store::new_existing(
        conn,
        config.table.name.clone(),
        semdoc::store::Store::arrow_schema(&config, &vec_fields),
        vec_fields.clone(),
        config.fields.clone(),
    );
    Ok((store, config))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let deploy = semdoc::config::DeploymentConfig::load(args.config.as_deref())?;
    deploy.apply_chunk_env();
    DEPLOY.set(deploy).ok();
    match args.cmd {
        Cmd::Init { schema, db } => init(schema, db).await,
        Cmd::Add { db, file, text, source, metas } => add(db, file, text, source, metas).await,
        Cmd::Query { db, text, mode, limit, filter, filter_sql } => query(db, text, mode, limit, filter, filter_sql).await,
        Cmd::Stats { db } => stats(db).await,
    }
}

async fn init(schema_path: String, db: String) -> Result<()> {
    let config = SchemaConfig::load(std::path::Path::new(&schema_path))?;
    // Probe the actual embedder dim once. If the config has explicit vector
    // fields, verify each auto_embed field's dim matches the embedder — a
    // mismatch is fatal at write time anyway; failing at init is kinder.
    let probe = match semdoc::embedding::Embedder::load(&deploy().embedding) {
        Ok(_) => semdoc::embedding::probe_dim_from_env().await,
        Err(_) => Err(anyhow::anyhow!("embedder unavailable")),
    };
    let embedder_dim = match &probe {
        Ok(d) => *d,
        Err(e) => {
            eprintln!("[init] warning: cannot probe embedder dim ({e:#}); skipping dim validation");
            1024
        }
    };
    if let Ok(actual) = probe {
        for vf in config.vector.fields.iter().filter(|v| v.auto_embed) {
            if vf.dim != actual {
                anyhow::bail!(
                    "vector column `{}`: configured dim {} != embedder output dim {} \
                     (embedding model: check Config.toml [embedding])",
                    vf.name,
                    vf.dim,
                    actual
                );
            }
        }
    }
    let vec_fields = config.effective_vector_fields(embedder_dim);
    std::fs::create_dir_all(&db)?;
    std::fs::copy(&schema_path, format!("{db}/schema.toml"))?;
    write_physical_fields(&db, &config)?;
    let conn = lancedb::connect(&db).execute().await?;
    Store::open(conn, &config, &vec_fields).await?;
    println!("initialized database at {db} ({} vector cols, {} user fields)",
        vec_fields.len(), config.fields.len());
    Ok(())
}

/// Persist the physical [fields] column set (name -> type) next to the
/// LanceDB data. Every later open compares the incoming schema against this
/// snapshot and fails fast on a mismatch, instead of silently misreading
/// rows written under a different schema.
pub fn write_physical_fields(db: &str, config: &SchemaConfig) -> Result<()> {
    let mut out = String::from("# Auto-generated by `semdoc init` — physical [fields] of this database.\n# Do not edit; regenerate by re-initializing (or run semdoc-migrate).\n[fields]\n");
    for (name, fc) in &config.fields {
        out.push_str(&format!("{name} = {{ type = \"{}\" }}\n", fc.r#type));
    }
    std::fs::write(format!("{db}/physical_fields.toml"), out)?;
    Ok(())
}

/// Check the incoming schema against `<db>/physical_fields.toml`. Missing
/// snapshot (pre-compat databases) → warn once and continue.
pub async fn check_schema_compat(db: &str, config: &SchemaConfig) -> Result<()> {
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

fn parse_metas(metas: &[String], config: &SchemaConfig) -> Result<serde_json::Map<String, serde_json::Value>> {
    let mut extra = serde_json::Map::new();
    for m in metas {
        let (k, v) = m
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--meta expects KEY=VALUE, got `{m}`"))?;
        let fc = config.fields.get(k)
            .ok_or_else(|| anyhow::anyhow!("--meta key `{k}` is not in schema [fields]"))?;
        let val = match fc.r#type.as_str() {
            "string" | "text" => serde_json::Value::String(v.to_string()),
            "bool" => serde_json::Value::Bool(v == "true" || v == "1"),
            "int64" | "timestamp" => serde_json::Value::Number(v.parse::<i64>()?.into()),
            "float32" => serde_json::Number::from_f64(v.parse::<f64>()?)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            "list<string>" => serde_json::Value::Array(
                v.split('|').map(|s| serde_json::Value::String(s.trim().to_string())).collect(),
            ),
            other => anyhow::bail!("unsupported meta type {other}"),
        };
        extra.insert(k.to_string(), val);
    }
    Ok(extra)
}

async fn add(db: String, file: Option<String>, text: Option<String>, source: String, metas: Vec<String>) -> Result<()> {
    let (store, config) = open_store(&db).await?;
    let text = match (file, text) {
        (Some(f), _) => std::fs::read_to_string(&f)?,
        (None, Some(t)) => t,
        _ => anyhow::bail!("--file or --text required"),
    };
    let mut extra = parse_metas(&metas, &config)?;
    // source_path/source_type are ordinary [fields] now; feed them through
    // extra when the schema declares them (Record-level fields remain for
    // backward compat with databases that have the fixed columns).
    if config.fields.contains_key("source_path") {
        extra.entry("source_path".to_string())
            .or_insert_with(|| serde_json::Value::String(source.clone()));
    }
    if config.fields.contains_key("source_type") {
        extra.entry("source_type".to_string())
            .or_insert_with(|| serde_json::Value::String("file".to_string()));
    }
    let embedder = semdoc::embedding::load_from_env()?;
    let doc_id = blake3::hash(text.as_bytes()).to_hex().to_string();
    // Graph plugin is needed both for replace (delete old mirror) and the
    // insert mirror below. Build it up front.
    let graph = semdoc::plugins::graph::build_graph_plugin(&config.plugins.graph).await?;
    // Replace-on-add: same semantics as MCP add_document — delete the old
    // version matching every replace_key before writing the new one.
    let replaced = replace_old_versions(&store, &config, &extra, &doc_id, graph.as_ref()).await?;
    let engine_inputs = InputDoc {
        id: doc_id.clone(),
        raw_text: text.clone(),
        source_path: source.clone(),
        source_type: "file".to_string(),
        language: Some(detect_language(&source, &text)),
        extra,
        vectors: Default::default(),
    };
    write_doc(&store, &embedder, engine_inputs).await?;
    store.optimize_indices().await?;
    // Mirror into the graph KB when the schema configures one. Entity
    // extraction is LLM-bound and slow; a failure here must not lose the
    // LanceDB write — degrade with a warning instead.
    if graph.name() != "none" {
        if let Err(e) = graph.insert(vec![(text, doc_id)]).await {
            eprintln!("[graph] insert degraded: {e:#}");
        }
    }
    println!(
        "added (id auto-hashed from content){}",
        replaced.as_ref().map(|old| format!(" — replaced old version {old}")).unwrap_or_default()
    );
    Ok(())
}

/// Replace-on-add shared logic: when the schema declares replace_key fields
/// and all of them are present in `extra`, delete every parent doc whose
/// values match ALL keys (AND), including leaf chunks and the lightrag
/// mirror. Returns the replaced doc id when exactly one version was swapped.
pub async fn replace_old_versions(
    store: &Store,
    config: &SchemaConfig,
    extra: &serde_json::Map<String, serde_json::Value>,
    new_doc_id: &str,
    graph: &dyn semdoc::plugins::graph::GraphPlugin,
) -> Result<Option<String>> {
    let rk = config.replace_key_fields();
    if rk.is_empty() {
        return Ok(None);
    }
    let mut conds = Vec::new();
    for k in &rk {
        match extra.get(*k).and_then(|v| v.as_str()) {
            Some(s) => conds.push(format!("{k} = '{}'", s.replace('\'', "''"))),
            None => return Ok(None), // keys not fully provided → append mode
        }
    }
    let sql = conds.join(" AND ");
    let olds = store.find_parents_by(&sql).await?;
    let mut replaced = None;
    for p in olds {
        if p.id == new_doc_id {
            continue;
        }
        store.delete_by_id(&p.id).await?;
        if graph.name() != "none" {
            if let Err(e) = graph.delete(&p.id).await {
                eprintln!("[replace] graph delete degraded for {}: {e:#}", p.id);
            }
        }
        replaced = Some(p.id);
    }
    Ok(replaced)
}

fn detect_language(path: &str, text: &str) -> String {
    if path.ends_with(".md") || path.ends_with(".markdown") || text.contains("## ") {
        "markdown".to_string()
    } else {
        "text".to_string()
    }
}

/// Write one document: parent row + leaf chunks, embedding each level for
/// every auto-embed vector column.
pub async fn write_doc(
    store: &Store,
    embedder: &semdoc::embedding::Embedder,
    doc: InputDoc,
) -> Result<()> {
    use std::collections::HashMap;
    let doc_id = doc.id.clone();
    let language = doc.language.clone();
    let source_path = doc.source_path.clone();

    // Parent record + vectors
    let mut parent_vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut leaf_vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let leaves_src = semdoc::chunker::chunk_for(&doc.raw_text, &doc_id, language.as_deref(), &source_path);

    for vf in &store.vector_fields {
        if !vf.auto_embed {
            continue; // pre-embedded only; covered by doc.vectors
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
            anyhow::bail!("vector column `{}`: source text field `{}` is empty", vf.name, vf.source);
        }
        let parent_vec = embedder.encode_blocking(src_text).await?;
        if parent_vec.len() != vf.dim {
            anyhow::bail!("vector column `{}`: embedder dim {} != configured dim {}", vf.name, parent_vec.len(), vf.dim);
        }
        let mut leaves_vecs: Vec<Vec<f32>> = Vec::with_capacity(leaves_src.len());
        for c in &leaves_src {
            let v = embedder.encode_blocking(c.text.clone()).await?;
            if v.len() != vf.dim {
                anyhow::bail!("vector column `{}`: embedder dim mismatch on leaf", vf.name);
            }
            leaves_vecs.push(v);
        }
        parent_vectors.insert(vf.name.clone(), parent_vec);
        if let Some(first) = leaves_vecs.first().cloned() {
            leaf_vectors.insert(vf.name.clone(), first);
        }
        for (idx, lv) in leaves_vecs.iter().enumerate() {
            leaf_vectors.insert(format!("{}#{}", vf.name, idx), lv.clone());
        }
    }

    let parent = Record {
        id: doc_id.clone(),
        raw_text: doc.raw_text,
        chunk_level: 0,
        chunk_index: 0,
        parent_doc_id: None,
        source_path,
        source_type: doc.source_type,
        extra: doc.extra,
    };
    let leaves: Vec<Record> = leaves_src
        .iter()
        .map(|c| Record {
            id: c.chunk_id.clone(),
            raw_text: c.text.clone(),
            chunk_level: 1,
            chunk_index: c.chunk_index,
            parent_doc_id: Some(doc_id.clone()),
            source_path: parent.source_path.clone(),
            source_type: parent.source_type.clone(),
            extra: parent.extra.clone(),
        })
        .collect();

    // Write each row via the builder path (parent uses parent_vectors;
    // leaves use per-leaf vectors from the "#idx" map).
    let mut all_rows: Vec<(Record, HashMap<String, Vec<f32>>)> = vec![(parent.clone(), parent_vectors.clone())];
    for (i, l) in leaves.iter().enumerate() {
        let mut lv: HashMap<String, Vec<f32>> = HashMap::new();
        for vf in &store.vector_fields {
            if let Some(v) = leaf_vectors.get(&format!("{}#{}", vf.name, i)) {
                lv.insert(vf.name.clone(), v.clone());
            } else if let Some(v) = parent_vectors.get(&vf.name) {
                lv.insert(vf.name.clone(), v.clone());
            }
        }
        all_rows.push((l.clone(), lv));
    }
    let refs: Vec<(&Record, &std::collections::HashMap<String, Vec<f32>>)> =
        all_rows.iter().map(|(r, v)| (r, v)).collect();
    let batch = store.build_batch_public(&refs)?;
    store.write_batch(batch).await?;
    Ok(())
}

async fn query(
    db: String,
    text: String,
    mode: String,
    limit: usize,
    filter: Option<String>,
    filter_sql: Option<String>,
) -> Result<()> {
    let (store, config) = open_store(&db).await?;
    let embedder = semdoc::embedding::Embedder::load(&deploy().embedding)?;
    let reranker = match semdoc::reranker::Reranker::load(&deploy().rerank) {
        Ok(r) => Some(std::sync::Arc::new(r)),
        Err(e) => {
            eprintln!("[rerank] unavailable: {e:#}");
            None
        }
    };
    let engine = Engine { store, embedder, reranker, config };

    let sql = match (filter, filter_sql) {
        (Some(f), _) => Some(semdoc::query::Pred::from_json(
            &serde_json::from_str::<serde_json::Value>(&f)?,
        )?
        .compile(&engine.config)?),
        (None, Some(s)) => Some(s),
        _ => None,
    };
    if let Some(s) = &sql {
        eprintln!("[filter] compiled: {s}");
    }

    let results = match mode.as_str() {
        "semantic" => engine.query_semantic(&text, limit, sql.as_deref(), ExpandTo::Chunk).await?,
        "parent" => engine.query_semantic(&text, limit, sql.as_deref(), ExpandTo::Parent).await?,
        "fts" => engine.query_fts(&text, limit, sql.as_deref()).await?,
        "rerank" => engine.query_reranked(&text, limit, sql.as_deref()).await?,
        other => anyhow::bail!("unknown mode `{other}` (semantic|parent|fts|rerank)"),
    };
    for r in &results {
        println!("{}", serde_json::to_string_pretty(&record_json(r))?);
        println!("---");
    }
    println!("[{} results]", results.len());
    Ok(())
}

async fn stats(db: String) -> Result<()> {
    let (store, _config) = open_store(&db).await?;
    println!("rows: {}", store.count().await?);
    Ok(())
}
