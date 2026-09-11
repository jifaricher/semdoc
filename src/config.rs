// SPDX-License-Identifier: MIT OR Apache-2.0
//! Deployment configuration — separate from schema.toml (library semantics).
//!
//! schema.toml travels WITH the database; the deployment config describes
//! the ENVIRONMENT (embedder endpoint, reranker endpoint, listen address,
//! chunk sizes) and must NOT travel with the database.
//!
//! Resolution priority (highest wins):
//!   1. CLI flags (handled by callers)
//!   2. Environment variables (`SEMDOC_*`, full backward compat with the
//!      env-only era)
//!   3. `Config.toml` (path via `--config`, `SEMDOC_CONFIG`, or
//!      `./Config.toml`)
//!   4. Built-in defaults
//!
//! Secrets never live in the TOML: string fields named `*_env` name an
//! environment variable whose value is the secret.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
// NOT deny_unknown_fields: a single Config.toml may also carry the schema
// sections ([table]/[vector]/[fields]/[plugins]) as a draft — those are
// parsed by SchemaConfig from schema.toml and intentionally ignored here.
pub struct DeploymentConfig {
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub rerank: RerankConfig,
    #[serde(default)]
    pub chunk: ChunkConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub tls: TlsConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
#[serde(tag = "backend")]
pub enum EmbeddingConfig {
    Onnx {
        #[serde(default)]
        dir: Option<String>,
    },
    Http {
        url: String,
        model: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_timeout_secs")]
        timeout_secs: u64,
    },
}

fn default_timeout_secs() -> u64 {
    60
}

impl Default for EmbeddingConfig {
    /// Default backend is `http` (parity with semrag deployments where the
    /// embedder runs as a separate HTTP service). url/model default empty —
    /// `validate` reports them as a clear config error rather than falling
    /// back to onnx.
    fn default() -> Self {
        EmbeddingConfig::Http {
            url: String::new(),
            model: String::new(),
            api_key_env: None,
            timeout_secs: default_timeout_secs(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
#[serde(tag = "backend")]
pub enum RerankConfig {
    Onnx {
        #[serde(default)]
        dir: Option<String>,
    },
    Tei {
        endpoint: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_timeout_secs")]
        timeout_secs: u64,
    },
    Openai {
        endpoint: String,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_timeout_secs")]
        timeout_secs: u64,
    },
    /// Rerank disabled — queries degrade to plain semantic.
    None,
}

impl Default for RerankConfig {
    /// Default backend is `tei`-over-HTTP (legacy env value `http` aliased
    /// here). endpoint defaults empty — `validate` errors clearly when unset.
    fn default() -> Self {
        RerankConfig::Tei {
            endpoint: String::new(),
            api_key_env: None,
            timeout_secs: 120,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct ChunkConfig {
    /// Target chunk size in chars. None = built-in default (512, clamped).
    #[serde(default)]
    pub size: Option<usize>,
    #[serde(default)]
    pub overlap: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default)]
    pub listen: Option<String>,
    /// Require `Authorization: Bearer <$token_env>`.
    #[serde(default)]
    pub token_env: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    /// Accept invalid certs (self-signed endpoints). Default false.
    #[serde(default)]
    pub insecure: bool,
}

impl DeploymentConfig {
    /// Discover + load + merge env overrides.
    ///
    /// File discovery order:
    ///   1. explicit `path` argument
    ///   2. `SEMDOC_CONFIG` env var
    ///   3. `./Config.toml`
    /// Missing file in cases 2/3 (and all of 1-3 absent) → defaults + env.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let file_path = path
            .map(String::from)
            .or_else(|| std::env::var("SEMDOC_CONFIG").ok())
            .unwrap_or_else(|| "Config.toml".to_string());
        let base = match std::fs::read_to_string(&file_path) {
            Ok(raw) => {
                let cfg: DeploymentConfig = toml::from_str(&raw).map_err(|e| {
                    anyhow::anyhow!("parse deployment config {file_path}: {e}")
                })?;
                cfg
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if path.is_some() || std::env::var("SEMDOC_CONFIG").is_ok() {
                    // Explicitly requested file must exist.
                    anyhow::bail!("deployment config not found: {file_path}");
                }
                DeploymentConfig::default()
            }
            Err(e) => anyhow::bail!("read deployment config {file_path}: {e}"),
        };
        let mut cfg = base;
        cfg.apply_env_overrides()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Env vars override file values. Every `SEMDOC_*` var from the
    /// env-only era keeps working — with no config file present, env alone
    /// fully configures the deployment (backward compatible).
    fn apply_env_overrides(&mut self) -> anyhow::Result<()> {
        // --- embedding ---
        // No backend named anywhere + no [embedding] in file → default http
        // (parity with semrag deployments). onnx requires the explicit
        // backend=onnx (or SEMDOC_EMBEDDER_BACKEND=onnx).
        if std::env::var("SEMDOC_EMBEDDER_BACKEND").is_err()
            && matches!(self.embedding, EmbeddingConfig::Onnx { .. })
            && std::env::var("SEMDOC_EMBEDDER_DIR").is_err()
        {
            self.embedding = EmbeddingConfig::default();
        }
        if let Ok(backend) = std::env::var("SEMDOC_EMBEDDER_BACKEND") {
            match backend.to_ascii_lowercase().as_str() {
                "http" => {
                    let url = std::env::var("SEMDOC_EMBEDDER_HTTP_URL")
                        .map_err(|_| anyhow::anyhow!("SEMDOC_EMBEDDER_HTTP_URL required for SEMDOC_EMBEDDER_BACKEND=http"))?;
                    let model = std::env::var("SEMDOC_EMBEDDER_HTTP_MODEL")
                        .map_err(|_| anyhow::anyhow!("SEMDOC_EMBEDDER_HTTP_MODEL required for SEMDOC_EMBEDDER_BACKEND=http"))?;
                    self.embedding = EmbeddingConfig::Http {
                        url,
                        model,
                        api_key_env: std::env::var("SEMDOC_EMBEDDER_HTTP_API_KEY_ENV").ok(),
                        timeout_secs: parse_env_num("SEMDOC_EMBEDDER_HTTP_TIMEOUT", 60)?,
                    };
                    // Legacy inline api key: treat as env-var indirection.
                    if let Ok(k) = std::env::var("SEMDOC_EMBEDDER_HTTP_API_KEY") {
                        if !k.is_empty() {
                            std::env::set_var("SEMDOC_EMBEDDER_HTTP_API_KEY_INLINE", k);
                            if let EmbeddingConfig::Http { api_key_env, .. } = &mut self.embedding {
                                *api_key_env = Some("SEMDOC_EMBEDDER_HTTP_API_KEY_INLINE".into());
                            }
                        }
                    }
                }
                "onnx" => {
                    self.embedding = EmbeddingConfig::Onnx {
                        dir: std::env::var("SEMDOC_EMBEDDER_DIR").ok(),
                    };
                }
                other => anyhow::bail!("unknown SEMDOC_EMBEDDER_BACKEND `{other}`"),
            }
        } else {
            // File may have specified http; legacy env vars still fill gaps.
            if let EmbeddingConfig::Http { api_key_env, timeout_secs, .. } = &mut self.embedding {
                if api_key_env.is_none() {
                    if let Ok(k) = std::env::var("SEMDOC_EMBEDDER_HTTP_API_KEY") {
                        if !k.is_empty() {
                            std::env::set_var("SEMDOC_EMBEDDER_HTTP_API_KEY_INLINE", k);
                            *api_key_env = Some("SEMDOC_EMBEDDER_HTTP_API_KEY_INLINE".into());
                        }
                    }
                }
                if *timeout_secs == default_timeout_secs() {
                    *timeout_secs = parse_env_num("SEMDOC_EMBEDDER_HTTP_TIMEOUT", default_timeout_secs())?;
                }
            }
        }

        // --- rerank ---
        // Only override when the env explicitly names a backend; otherwise
        // the file's [rerank] stands.
        if std::env::var("SEMDOC_RERANKER_BACKEND").is_err()
            && matches!(self.rerank, RerankConfig::Onnx { .. })
            && std::env::var("SEMDOC_RERANKER_DIR").is_err()
        {
            self.rerank = RerankConfig::default();
        }
        if let Ok(backend) = std::env::var("SEMDOC_RERANKER_BACKEND") {
            self.rerank = match backend.to_ascii_lowercase().as_str() {
                "onnx" => RerankConfig::Onnx { dir: std::env::var("SEMDOC_RERANKER_DIR").ok() },
                // `tei` is the canonical name; `http` is the legacy env-era
                // alias (semrag's SEMDOC_RERANKER_BACKEND=http meant TEI).
                "tei" | "http" => RerankConfig::Tei {
                    endpoint: std::env::var("SEMDOC_RERANKER_HTTP_URL")
                        .map_err(|_| anyhow::anyhow!("SEMDOC_RERANKER_HTTP_URL required for SEMDOC_RERANKER_BACKEND=tei"))?,
                    api_key_env: inline_or_named_key("SEMDOC_RERANKER_HTTP_API_KEY"),
                    timeout_secs: parse_env_num("SEMDOC_RERANKER_HTTP_TIMEOUT", 120)?,
                },
                "openai" => RerankConfig::Openai {
                    endpoint: std::env::var("SEMDOC_RERANKER_HTTP_URL")
                        .map_err(|_| anyhow::anyhow!("SEMDOC_RERANKER_HTTP_URL required for SEMDOC_RERANKER_BACKEND=openai"))?,
                    model: std::env::var("SEMDOC_RERANKER_MODEL").ok(),
                    api_key_env: inline_or_named_key("SEMDOC_RERANKER_HTTP_API_KEY"),
                    timeout_secs: parse_env_num("SEMDOC_RERANKER_HTTP_TIMEOUT", 120)?,
                },
                other => anyhow::bail!("unknown SEMDOC_RERANKER_BACKEND `{other}`"),
            };
        }

        // --- chunk ---
        if let Ok(v) = std::env::var("SEMDOC_CHUNK_SIZE") {
            self.chunk.size = Some(v.parse().map_err(|_| anyhow::anyhow!("bad SEMDOC_CHUNK_SIZE `{v}`"))?);
        }
        if let Ok(v) = std::env::var("SEMDOC_CHUNK_OVERLAP") {
            self.chunk.overlap = Some(v.parse().map_err(|_| anyhow::anyhow!("bad SEMDOC_CHUNK_OVERLAP `{v}`"))?);
        }

        // --- server ---
        if let Ok(v) = std::env::var("SEMDOC_LISTEN") {
            self.server.listen = Some(v);
        }

        // --- tls ---
        if let Ok(v) = std::env::var("SEMDOC_TLS_INSECURE") {
            self.tls.insecure = v == "1" || v.eq_ignore_ascii_case("true");
        }
        Ok(())
    }

    fn validate(&self) -> anyhow::Result<()> {
        // *_env indirections must name vars that are SET (fail fast, with a
        // clear message, instead of a 401 at first query).
        let check = |name: &str, var: &Option<String>| -> anyhow::Result<()> {
            if let Some(v) = var {
                if std::env::var(v).is_err() {
                    anyhow::bail!("{name}: referenced env var `{v}` is not set");
                }
            }
            Ok(())
        };
        match &self.embedding {
            EmbeddingConfig::Http { api_key_env, .. } => {
                check("embedding.api_key_env", api_key_env)?
            }
            EmbeddingConfig::Onnx { dir } => {
                if !cfg!(feature = "onnx") {
                    anyhow::bail!(
                        "embedding backend onnx requires building with feature `onnx`"
                    );
                }
                if let Some(d) = dir {
                    if !Path::new(d).join("model.onnx").exists() {
                        anyhow::bail!("embedding.dir: no model.onnx under `{d}`");
                    }
                }
            }
        }
        match &self.rerank {
            RerankConfig::Tei { endpoint, api_key_env, .. }
            | RerankConfig::Openai { endpoint, api_key_env, .. } => {
                if endpoint.is_empty() {
                    anyhow::bail!(
                        "rerank backend=http requires endpoint \
                         (set [rerank] in Config.toml or SEMDOC_RERANKER_HTTP_URL)"
                    );
                }
                check("rerank.api_key_env", api_key_env)?
            }
            RerankConfig::Onnx { dir } => {
                if !cfg!(feature = "onnx") {
                    anyhow::bail!("rerank backend onnx requires building with feature `onnx`");
                }
                if let Some(d) = dir {
                    if !Path::new(d).join("model.onnx").exists() {
                        anyhow::bail!("rerank.dir: no model.onnx under `{d}`");
                    }
                }
            }
            RerankConfig::None => {}
        }
        if let Some(t) = &self.server.token_env {
            if std::env::var(t).is_err() {
                anyhow::bail!("server.token_env: referenced env var `{t}` is not set");
            }
        }
        if let Some(size) = self.chunk.size {
            if !(64..=8192).contains(&size) {
                anyhow::bail!("chunk.size must be within [64, 8192]");
            }
        }
        Ok(())
    }

    /// Resolve the server bearer token (if any) into an owned value.
    pub fn server_token(&self) -> Option<String> {
        self.server
            .token_env
            .as_ref()
            .and_then(|e| std::env::var(e).ok())
    }

    /// Push chunk settings into the env vars the chunker reads (it resolves
    /// them per call, so this propagates to every chunk_for invocation).
    pub fn apply_chunk_env(&self) {
        if let Some(s) = self.chunk.size {
            std::env::set_var("SEMDOC_CHUNK_SIZE", s.to_string());
        }
        if let Some(o) = self.chunk.overlap {
            std::env::set_var("SEMDOC_CHUNK_OVERLAP", o.to_string());
        }
    }
}

fn parse_env_num(name: &str, default: u64) -> anyhow::Result<u64> {
    match std::env::var(name) {
        Ok(v) => v.parse().map_err(|_| anyhow::anyhow!("bad {name} `{v}`")),
        Err(_) => Ok(default),
    }
}

/// Legacy inline API key env var: if set, route it through an env-var
/// indirection (the value IS the key; we park it under a synthetic var name
/// so the rest of the code uniformly reads keys via `*_env` indirection).
fn inline_or_named_key(inline_var: &str) -> Option<String> {
    if let Ok(k) = std::env::var(inline_var) {
        if !k.is_empty() {
            return Some(inline_var.to_string());
        }
    }
    None
}
