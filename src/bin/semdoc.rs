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

/// Remote backend: talk to a running semdoc-server over its REST API
/// instead of opening the LanceDB directory directly.
#[derive(Clone)]
struct Remote {
    base: String,
    token: Option<String>,
    http: reqwest::blocking::Client,
}

impl Remote {
    fn new(base: &str, token: Option<String>) -> Result<Self> {
        Ok(Self {
            base: base.trim_end_matches('/').to_string(),
            token,
            http: semdoc::tls::apply_blocking(reqwest::blocking::Client::builder())
                .timeout(std::time::Duration::from_secs(300))
                .build()?,
        })
    }

    fn req(&self, method: reqwest::Method, path: &str) -> reqwest::blocking::RequestBuilder {
        let mut r = self.http.request(method, format!("{}{path}", self.base));
        if let Some(t) = &self.token {
            r = r.bearer_auth(t);
        }
        r
    }

    fn check(resp: reqwest::blocking::Response) -> Result<serde_json::Value> {
        let status = resp.status();
        let body = resp.text()?;
        if !status.is_success() {
            anyhow::bail!("server {} : {body}", status);
        }
        Ok(serde_json::from_str(&body).unwrap_or(serde_json::Value::Null))
    }

    fn add(&self, text: &str, source_path: &str, meta: serde_json::Map<String, serde_json::Value>) -> Result<serde_json::Value> {
        Self::check(self.req(reqwest::Method::POST, "/documents")
            .json(&serde_json::json!({"text": text, "source_path": source_path, "meta": meta}))
            .send()?)
    }

    fn query(&self, endpoint: &str, body: serde_json::Value) -> Result<Vec<serde_json::Value>> {
        let v = Self::check(self.req(reqwest::Method::POST, endpoint).json(&body).send()?)?;
        Ok(v.get("documents")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default())
    }

    fn delete(&self, id: &str) -> Result<()> {
        let resp = self.req(reqwest::Method::POST, "/documents/delete")
            .json(&serde_json::json!({"id": id}))
            .send()?;
        let status = resp.status();
        let body = resp.text()?;
        if status == reqwest::StatusCode::NOT_FOUND {
            anyhow::bail!("document not found: {id}");
        }
        if !status.is_success() {
            anyhow::bail!("server {status} : {body}");
        }
        Ok(())
    }

    fn stats(&self) -> Result<serde_json::Value> {
        Self::check(self.req(reqwest::Method::GET, "/stats").send()?)
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Initialize a database from a schema config
    Init {
        /// Path to schema.toml (or omit and use --template)
        #[arg(short, long)]
        schema: Option<String>,
        /// Built-in template: generic | code-search | paper-library | kernel-docs
        #[arg(long)]
        template: Option<String>,
        /// Database directory
        #[arg(short, long)]
        db: String,
    },
    /// Add a document
    Add {
        #[arg(short, long)]
        db: Option<String>,
        /// Connect to a running semdoc-server instead of the local db
        #[arg(long)]
        server: Option<String>,
        /// Bearer token for --server (default: $SEMDOC_TOKEN)
        #[arg(long)]
        token: Option<String>,
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
        db: Option<String>,
        /// Connect to a running semdoc-server instead of the local db
        #[arg(long)]
        server: Option<String>,
        /// Bearer token for --server (default: $SEMDOC_TOKEN)
        #[arg(long)]
        token: Option<String>,
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
        /// Filter as raw SQL (trusted input only; local db only)
        #[arg(long)]
        filter_sql: Option<String>,
    },
    /// Delete a document (cascades: parent + leaf chunks + lightrag mirror)
    Delete {
        #[arg(short, long)]
        db: Option<String>,
        /// Connect to a running semdoc-server instead of the local db
        #[arg(long)]
        server: Option<String>,
        /// Bearer token for --server (default: $SEMDOC_TOKEN)
        #[arg(long)]
        token: Option<String>,
        /// Document id (parent or leaf chunk; a chunk id deletes its parent)
        #[arg(short, long)]
        id: String,
        /// Skip the existence check (idempotent delete; no error when absent)
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Build/refresh vector + scalar indexes for an existing database.
    /// IVF vector indexes need data to train centroids — run this AFTER
    /// bulk imports (init-time index creation on an empty table is skipped
    /// by lancedb). Also compacts fragmented index segments.
    Reindex {
        #[arg(short, long)]
        db: String,
        /// Rebuild even if an index already exists (drop + recreate)
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Run environment / database health checks with fix hints
    Doctor {
        /// Database directory to check (schema, indexes, row counts)
        #[arg(short, long)]
        db: Option<String>,
        /// Also probe a running server (REST /stats + /health)
        #[arg(long)]
        server: Option<String>,
        /// Bearer token for --server (default: $SEMDOC_TOKEN)
        #[arg(long)]
        token: Option<String>,
    },
    /// Show database stats
    Stats {
        #[arg(short, long)]
        db: Option<String>,
        /// Connect to a running semdoc-server instead of the local db
        #[arg(long)]
        server: Option<String>,
        /// Bearer token for --server (default: $SEMDOC_TOKEN)
        #[arg(long)]
        token: Option<String>,
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
        Cmd::Init { schema, db, template } => init(schema, template, db).await,
        Cmd::Add { db, server, token, file, text, source, metas } => add(db, server, token, file, text, source, metas).await,
        Cmd::Query { db, server, token, text, mode, limit, filter, filter_sql } => query(db, server, token, text, mode, limit, filter, filter_sql).await,
        Cmd::Delete { db, server, token, id, force } => delete(db, server, token, id, force).await,
        Cmd::Reindex { db, force } => reindex(db, force).await,
        Cmd::Doctor { db, server, token } => doctor(db, server, token).await,
        Cmd::Stats { db, server, token } => stats(db, server, token).await,
    }
}


async fn init(schema_path: Option<String>, template: Option<String>, db: String) -> Result<()> {
    let schema_path = match (schema_path, template) {
        (Some(p), None) => p,
        (None, Some(t)) => {
            let content = semdoc::schema::template_schema_content(&t)?;
            let path = format!("{db}.template.schema.toml");
            std::fs::create_dir_all(&db)?;
            std::fs::write(&path, content)?;
            println!("template `{t}` written to {path}");
            path
        }
        (Some(_), Some(_)) => anyhow::bail!("--schema and --template are mutually exclusive"),
        (None, None) => anyhow::bail!("--schema <path> or --template <name> is required"),
    };
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

/// Health checks with fix hints. Every failure is actionable — this
/// encodes the operational pitfalls hit during development (embedder dim
/// mismatch, lightrag down, TLS CA, schema drift, missing indexes).
async fn doctor(db: Option<String>, server: Option<String>, token: Option<String>) -> Result<()> {
    let mut failures = 0;
    let ok = |name: &str, detail: &str| println!("✓ {name}: {detail}");
    let bad = |name: &str, detail: &str| {
        println!("✗ {name}: {detail}");
    };
    let warn = |name: &str, detail: &str| println!("⚠ {name}: {detail}");

    // 1. deployment config
    let deploy_path = std::env::var("SEMDOC_CONFIG").unwrap_or_else(|_| "./Config.toml".into());
    match semdoc::config::DeploymentConfig::load(None) {
        Ok(cfg) => {
            ok("config", &format!("{deploy_path} loads"));
            // 2. embedder probe (one real encode)
            match semdoc::embedding::Embedder::load(&cfg.embedding) {
                Ok(e) => match e.encode_blocking("doctor probe".into()).await {
                    Ok(v) => ok("embedder", &format!("probe OK, dim={}", v.len())),
                    Err(err) => {
                        bad("embedder", &format!("probe failed: {err:#}"));
                        failures += 1;
                    }
                },
                Err(err) => {
                    bad("embedder", &format!("{err:#}"));
                    failures += 1;
                }
            }
            // 3. reranker (optional — absence is fine)
            match semdoc::reranker::Reranker::load(&cfg.rerank) {
                Ok(_) => ok("reranker", "configured and loadable"),
                Err(e) => {
                    let m = format!("{e:#}");
                    if m.contains("disabled by config") || m.contains("none") {
                        warn("reranker", "disabled — queries degrade to ANN order");
                    } else {
                        bad("reranker", &m);
                        failures += 1;
                    }
                }
            }
        }
        Err(e) => {
            bad("config", &format!("{e:#}"));
            failures += 1;
        }
    }

    // 4. database checks
    if let Some(db) = &db {
        match open_store(db).await {
            Ok((store, config)) => {
                ok("schema", &format!(
                    "{} fields, {} vector cols, table `{}`",
                    config.fields.len(),
                    store.vector_fields.len(),
                    store.table
                ));
                let rows = store.count().await?;
                ok("lancedb", &format!("{rows} rows"));
                for vf in &store.vector_fields {
                    if vf.index == "none" && rows >= 50_000 {
                        warn(
                            &format!("vector index `{}`", vf.name),
                            &format!(
                                "brute-force over {rows} rows — set index=\"ivf_flat\" in schema.toml and run `semdoc reindex --db {db}`"
                            ),
                        );
                    }
                }
            }
            Err(e) => {
                bad("database", &format!("{db}: {e:#}"));
                failures += 1;
            }
        }
    } else if db.is_none() && server.is_none() {
        warn("database", "no --db given; skipping db checks");
    }

    // 5. server probe
    if let Some(url) = &server {
        let tok = token.clone().or_else(|| std::env::var("SEMDOC_TOKEN").ok());
        match reqwest::blocking::Client::new()
            .get(format!("{url}/stats"))
            .apply_bearer(&tok)
            .timeout(std::time::Duration::from_secs(5))
            .send()
        {
            Ok(r) if r.status().is_success() => ok("server", &format!("{url} reachable")),
            Ok(r) => {
                bad("server", &format!("{url} HTTP {}", r.status()));
                failures += 1;
            }
            Err(e) => {
                bad("server", &format!("{url}: {e}"));
                failures += 1;
            }
        }
    }

    // 6. graph plugin (from schema of the db if provided)
    if let Some(db) = &db {
        let cfg_path = format!("{db}/schema.toml");
        if let Ok(config) = SchemaConfig::load(std::path::Path::new(&cfg_path)) {
            let graph = semdoc::plugins::graph::build_graph_plugin(&config.plugins.graph).await?;
            if graph.name() == "none" {
                warn("graph", "not configured — query_graph degrades to semantic");
            } else {
                match graph.health().await {
                    Ok(()) => ok("graph", &format!("{} backend healthy", graph.name())),
                    Err(e) => {
                        bad("graph", &format!("{}: {e:#}", graph.name()));
                        println!("    hint: ./scripts/start-lightrag-server.sh --bg");
                        failures += 1;
                    }
                }
            }
        }
    }

    if failures == 0 {
        println!("doctor: all checks passed");
    } else {
        anyhow::bail!("doctor: {failures} check(s) failed");
    }
    Ok(())
}

trait Bearer {
    fn apply_bearer(self, token: &Option<String>) -> reqwest::blocking::RequestBuilder;
}
impl Bearer for reqwest::blocking::RequestBuilder {
    fn apply_bearer(self, token: &Option<String>) -> reqwest::blocking::RequestBuilder {
        match token {
            Some(t) => self.bearer_auth(t),
            None => self,
        }
    }
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

async fn add(
    db: Option<String>,
    server: Option<String>,
    token: Option<String>,
    file: Option<String>,
    text: Option<String>,
    source: String,
    metas: Vec<String>,
) -> Result<()> {
    if let Some(url) = server {
        let remote = Remote::new(&url, token.or_else(|| std::env::var("SEMDOC_TOKEN").ok()))?;
        let text = match (file, text) {
            (Some(f), _) => std::fs::read_to_string(&f)?,
            (None, Some(t)) => t,
            _ => anyhow::bail!("--file or --text required"),
        };
        let meta: serde_json::Map<String, serde_json::Value> = metas
            .iter()
            .filter_map(|m| m.split_once('=').map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string()))))
            .collect();
        remote.add(&text, &source, meta)?;
        println!("added (id auto-hashed from content)");
        return Ok(());
    }
    let db = db.expect("clap: db or server required");
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
            if let Err(e) = graph.delete_with_retry(&p.id, 3).await {
                eprintln!("[replace] graph delete degraded for {}: {e:#}", p.id);
            }
        }
        replaced = Some(p.id);
    }
    Ok(replaced)
}

async fn delete(
    db: Option<String>,
    server: Option<String>,
    token: Option<String>,
    id: String,
    force: bool,
) -> Result<()> {
    if let Some(url) = server {
        let remote = Remote::new(&url, token.or_else(|| std::env::var("SEMDOC_TOKEN").ok()))?;
        remote.delete(&id)?;
        println!("deleted {id}");
        return Ok(());
    }
    let db = db.expect("clap: db or server required");
    let (store, config) = open_store(&db).await?;
    if id.is_empty() {
        anyhow::bail!("--id is required");
    }
    // Existence check (skippable with --force): chunk id → parent id.
    let target = if force {
        id.clone()
    } else {
        match store.get_by_id(&id).await? {
            None => anyhow::bail!("document not found: {id} (use --force to delete blindly)"),
            Some(rec) => rec.parent_doc_id.clone().unwrap_or(rec.id),
        }
    };
    store.delete_by_id(&target).await?;
    // lightrag mirror: busy-retried (3 attempts, 10s/20s backoff). Failure
    // leaves graph residue — reported, and the same command can be re-run
    // later to clean up (delete is idempotent on both sides).
    let graph = semdoc::plugins::graph::build_graph_plugin(&config.plugins.graph).await?;
    if graph.name() != "none" {
        match graph.delete_with_retry(&target, 3).await {
            Ok(()) => println!("deleted {target} (lancedb + lightrag)"),
            Err(e) => {
                eprintln!("[graph] delete degraded after retries: {e:#}");
                println!("deleted {target} (lancedb) — lightrag residue possible, re-run to retry");
            }
        }
    } else {
        println!("deleted {target} (lancedb; no graph backend configured)");
    }
    Ok(())
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

#[allow(clippy::too_many_arguments)]
async fn query(
    db: Option<String>,
    server: Option<String>,
    token: Option<String>,
    text: String,
    mode: String,
    limit: usize,
    filter: Option<String>,
    filter_sql: Option<String>,
) -> Result<()> {
    // Remote: Mongo-style filter JSON is sent as-is (server compiles it
    // against the db schema). Raw SQL never crosses the network.
    if let Some(url) = server {
        if filter_sql.is_some() {
            anyhow::bail!("--filter_sql is not supported with --server (trusted-input only, local db)");
        }
        let remote = Remote::new(&url, token.or_else(|| std::env::var("SEMDOC_TOKEN").ok()))?;
        let filter_json: Option<serde_json::Value> = filter
            .as_ref()
            .map(|f| serde_json::from_str(f))
            .transpose()?;
        let endpoint = match mode.as_str() {
            "semantic" => "/query/semantic",
            "parent" => "/query/semantic",
            "fts" => "/query/text",
            "rerank" => "/query/reranked",
            other => anyhow::bail!("unknown mode `{other}` (semantic|parent|fts|rerank)"),
        };
        let mut body = serde_json::json!({"text": text, "limit": limit});
        if let Some(f) = &filter_json {
            body["filter"] = f.clone();
        }
        if mode == "parent" {
            body["expand_to"] = serde_json::json!("parent");
        }
        let docs = remote.query(endpoint, body)?;
        for r in &docs {
            println!("{}", serde_json::to_string_pretty(r)?);
            println!("---");
        }
        return Ok(());
    }
    let db = db.expect("clap: db or server required");
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

async fn stats(db: Option<String>, server: Option<String>, token: Option<String>) -> Result<()> {
    if let Some(url) = server {
        let remote = Remote::new(&url, token.or_else(|| std::env::var("SEMDOC_TOKEN").ok()))?;
        println!("stats: {}", remote.stats()?);
        return Ok(());
    }
    let db = db.expect("clap: db or server required");
    let (store, _config) = open_store(&db).await?;
    let rows = store.count().await?;
    println!("rows: {rows}");
    // Nudge before the brute-force scan cliff: ANN with index="none" is
    // linear over all rows; IVF turns it into ~sqrt. 50k is where the
    // p50 latency becomes noticeable (~100ms+ per query on CPU).
    if rows >= 50_000 {
        for vf in &store.vector_fields {
            if vf.index == "none" {
                eprintln!(
                    "[hint] vector column `{}` has index=\"none\" with {rows} rows — queries \
brute-force scan the table. Set index=\"ivf_flat\" in schema.toml and run: \
semdoc reindex --db {db}",
                    vf.name
                );
            }
        }
    }
    Ok(())
}

/// Build vector indexes for columns configured with ivf_flat/ivf_pq, then
/// optimize (compaction + FTS/BTree delta indexing). Safe to re-run.
async fn reindex(db: String, force: bool) -> Result<()> {
    let (store, _config) = open_store(&db).await?;
    let t = store.conn.open_table(&store.table).execute().await?;
    let rows = store.count().await?;
    println!("reindexing {db} ({rows} rows)");
    let mut built = 0;
    for vf in &store.vector_fields {
        if vf.index == "none" {
            println!("  {}: no vector index configured (brute-force) — skipped", vf.name);
            continue;
        }
        if rows < 256 {
            println!(
                "  {}: only {rows} rows — IVF training needs >=256; skipping (brute force is faster at this size)",
                vf.name
            );
            continue;
        }
        let kind = vf.index.as_str();
        if force {
            // List and drop any existing index on this column first.
            let idx_names = t.list_indices().await?;
            for idx in idx_names {
                if idx.columns.len() == 1 && idx.columns[0] == vf.name {
                    println!("  {}: dropping existing index `{}`", vf.name, idx.name);
                    t.drop_index(&idx.name).await?;
                }
            }
        }
        let res = store.ensure_vector_indexes().await;
        match res {
            Ok(()) => {
                println!("  {}: {kind} index ready", vf.name);
                built += 1;
            }
            Err(e) => {
                let msg = format!("{e:#}");
                if msg.contains("already exists") {
                    println!("  {}: {kind} index already exists (use --force to rebuild)", vf.name);
                } else {
                    eprintln!("  {}: {kind} index FAILED: {msg}", vf.name);
                }
            }
        }
    }
    store.optimize_indices().await?;
    println!(
        "done: {built} vector index built/verified, optimize (compaction + FTS/BTree deltas) complete"
    );
    Ok(())
}
