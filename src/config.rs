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
    /// POST a JSON event here after every successful write (add / delete /
    /// update). Fire-and-forget: failures log to stderr, never fail the
    /// original operation.
    #[serde(default)]
    pub webhook_url: Option<String>,
    /// Which events to deliver. Default = all of `add`,`delete`,`update`.
    #[serde(default)]
    pub webhook_events: Option<Vec<String>>,
    /// Env var naming the HMAC-SHA256 secret for the
    /// `X-Semdoc-Signature: sha256=<hex>` header. Required for the
    /// receiver to authenticate the call — without a secret the receiver
    /// cannot tell semdoc from an intruder.
    #[serde(default)]
    pub webhook_secret_env: Option<String>,
}

impl ServerConfig {
    /// Whether a webhook event kind is enabled. `None` url → off.
    pub fn webhook_enabled(&self, event: &str) -> bool {
        match (&self.webhook_url, &self.webhook_events) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(_), Some(list)) => list.iter().any(|e| e == event),
        }
    }

    pub fn webhook_secret(&self) -> Option<String> {
        self.webhook_secret_env
            .as_ref()
            .and_then(|e| std::env::var(e).ok())
            .filter(|s| !s.is_empty())
    }
}

#[derive(Debug, Clone, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    /// Accept invalid certs (self-signed endpoints). Default false.
    /// Prefer `ca_bundle` when possible — this flag disables validation
    /// entirely, exposing Bearer tokens to interception.
    #[serde(default)]
    pub insecure: bool,
    /// Path to a PEM bundle with the company CA cert(s) to trust in
    /// addition to the system store (e.g. `/etc/semdoc/company-ca.pem`).
    /// Clients keep validating everything else. Empty = system store only.
    #[serde(default)]
    pub ca_bundle: Option<String>,
}

impl TlsConfig {
    /// Apply the config to the process-wide TLS environment so that every
    /// HTTP client (embedder / reranker / graph plugin) picks it up. This
    /// bridges the config-file setting to the shared `semdoc::tls` helper —
    /// env vars set explicitly by the operator still win (they are applied
    /// later by `apply_env_overrides` and re-checked here).
    pub fn apply_env(&self) {
        if self.insecure {
            std::env::set_var(semdoc_tls::ENV_INSECURE, "1");
        }
        if let Some(p) = &self.ca_bundle {
            std::env::set_var(semdoc_tls::ENV_CA_BUNDLE, p);
        }
    }
}

/// Minimal facade over the shared TLS helper so config code doesn't depend
/// on the whole `tls` module (env var names live there).
mod semdoc_tls {
    pub const ENV_INSECURE: &str = "SEMDOC_TLS_INSECURE";
    pub const ENV_CA_BUNDLE: &str = "SEMDOC_CA_BUNDLE";
}

impl DeploymentConfig {
    /// Discover + load + merge env overrides.
    ///
    /// File discovery order:
    ///   1. explicit `path` argument
    ///   2. `SEMDOC_CONFIG` env var
    ///   3. `./Config.toml`
    ///
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
        if let Ok(v) = std::env::var("SEMDOC_CA_BUNDLE") {
            if !v.is_empty() {
                self.tls.ca_bundle = Some(v);
            }
        }
        // Re-export the resolved TLS settings so every HTTP client (they read
        // the env via semdoc::tls) sees the Config.toml values too — the
        // config file is the default source, env vars can still override
        // because they were merged into self.tls above.
        self.tls.apply_env();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Env-var mutation is process-global; serialize every test that touches
    /// SEMDOC_* vars (and scrub them before each use).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const SCRUB: &[&str] = &[
        "SEMDOC_EMBEDDER_BACKEND",
        "SEMDOC_EMBEDDER_DIR",
        "SEMDOC_EMBEDDER_HTTP_URL",
        "SEMDOC_EMBEDDER_HTTP_MODEL",
        "SEMDOC_EMBEDDER_HTTP_API_KEY",
        "SEMDOC_EMBEDDER_HTTP_API_KEY_ENV",
        "SEMDOC_EMBEDDER_HTTP_TIMEOUT",
        "SEMDOC_RERANKER_BACKEND",
        "SEMDOC_RERANKER_DIR",
        "SEMDOC_RERANKER_HTTP_URL",
        "SEMDOC_RERANKER_HTTP_API_KEY",
        "SEMDOC_RERANKER_MODEL",
        "SEMDOC_RERANKER_HTTP_TIMEOUT",
        "SEMDOC_CHUNK_SIZE",
        "SEMDOC_CHUNK_OVERLAP",
        "SEMDOC_LISTEN",
        "SEMDOC_TLS_INSECURE",
        "SEMDOC_TEST_TOKEN",
        "SEMDOC_TEST_API_KEY",
        "SEMDOC_CONFIG",
    ];

    fn with_clean_env(f: impl FnOnce()) {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for k in SCRUB {
            std::env::remove_var(k);
        }
        f();
        for k in SCRUB {
            std::env::remove_var(k);
        }
    }

    fn parse(raw: &str) -> anyhow::Result<DeploymentConfig> {
        let mut cfg: DeploymentConfig = toml::from_str(raw)?;
        cfg.apply_env_overrides()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse + env-override without validate() — for configs whose defaults
    /// (empty rerank endpoint) fail validation.
    fn parse_unchecked(raw: &str) -> anyhow::Result<DeploymentConfig> {
        let mut cfg: DeploymentConfig = toml::from_str(raw)?;
        cfg.apply_env_overrides()?;
        Ok(cfg)
    }

    // --- TOML parsing / defaults ---

    #[test]
    fn empty_toml_gives_defaults() {
        with_clean_env(|| {
            let cfg = parse_unchecked("").unwrap();
            assert!(matches!(cfg.embedding, EmbeddingConfig::Http { ref url, .. } if url.is_empty()));
            assert!(matches!(cfg.rerank, RerankConfig::Tei { ref endpoint, .. } if endpoint.is_empty()));
            assert_eq!(cfg.chunk.size, None);
            assert_eq!(cfg.server.listen, None);
            assert!(!cfg.tls.insecure);
        });
    }

    #[test]
    #[cfg(feature = "onnx")] // validates the onnx backend path
    fn parse_onnx_and_none_backends() {
        // NOTE: apply_env_overrides() downgrades a file-specified
        // `backend = "onnx"` to default http unless SEMDOC_EMBEDDER_DIR is
        // set in the environment (current intended behavior — onnx from the
        // env-only era required the dir env var). validate() additionally
        // requires model.onnx to exist under the dir, so fabricate one.
        with_clean_env(|| {
            let dir = std::env::temp_dir().join(format!("semdoc-onnx-test-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.onnx"), b"fake").unwrap();
            // Setting BACKEND=onnx keeps onnx; the effective dir then comes
            // from SEMDOC_EMBEDDER_DIR (env wins over the file's dir).
            std::env::set_var("SEMDOC_EMBEDDER_BACKEND", "onnx");
            std::env::set_var("SEMDOC_EMBEDDER_DIR", dir.to_str().unwrap());
            let cfg = parse(
                r#"
[rerank]
backend = "none"
"#,
            )
            .unwrap();
            assert!(matches!(cfg.embedding, EmbeddingConfig::Onnx { .. }));
            assert!(matches!(cfg.rerank, RerankConfig::None));
            std::fs::remove_dir_all(&dir).ok();
        });
    }

    #[test]
    fn file_onnx_without_env_dir_is_downgraded_to_default_http() {
        with_clean_env(|| {
            let cfg = parse_unchecked(
                r#"
[embedding]
backend = "onnx"
dir = "/models/bge-m3"
"#,
            )
            .unwrap();
            assert!(matches!(cfg.embedding, EmbeddingConfig::Http { .. }));
        });
    }
    #[test]
    fn kebab_case_fields_accepted() {
        with_clean_env(|| {
            let cfg = parse_unchecked(
                r#"
[embedding]
backend = "http"
url = "http://embed:80/v1"
model = "bge-m3"
api_key_env = "SEMDOC_TEST_API_KEY"
timeout_secs = 5
"#,
            )
            .unwrap();
            match &cfg.embedding {
                EmbeddingConfig::Http { url, model, api_key_env, timeout_secs } => {
                    assert_eq!(url, "http://embed:80/v1");
                    assert_eq!(model, "bge-m3");
                    assert_eq!(api_key_env.as_deref(), Some("SEMDOC_TEST_API_KEY"));
                    assert_eq!(*timeout_secs, 5);
                }
                other => panic!("expected http embedder, got {other:?}"),
            }
        });
    }

    #[test]
    fn unknown_fields_rejected() {
        let err = toml::from_str::<DeploymentConfig>("[chunk]\nbogus = 1\n").unwrap_err();
        assert!(err.to_string().contains("unknown field"));
        let err = toml::from_str::<DeploymentConfig>("[server]\nbogus = 1\n").unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn schema_sections_are_ignored() {
        // A single Config.toml may also carry the schema sections as a draft.
        with_clean_env(|| {
            let cfg = parse_unchecked(
                r#"
[table]
name = "documents"

[vector]
fields = []

[fields]
category = { type = "string" }

[plugins]
"#,
            )
            .unwrap();
            assert_eq!(cfg.chunk.size, None);
        });
    }

    // --- validate() ---

    #[test]
    fn validate_rejects_empty_rerank_endpoint() {
        with_clean_env(|| {
            let cfg = DeploymentConfig::default();
            assert!(cfg.validate().is_err());
        });
    }

    #[test]
    fn validate_accepts_rerank_none_with_empty_endpoint() {
        with_clean_env(|| {
            let cfg = DeploymentConfig { rerank: RerankConfig::None, ..Default::default() };
            cfg.validate().unwrap();
        });
    }

    #[test]
    fn validate_rejects_unset_token_env() {
        with_clean_env(|| {
            let cfg = DeploymentConfig {
                rerank: RerankConfig::None,
                server: ServerConfig { token_env: Some("SEMDOC_TEST_TOKEN".into()), listen: None, ..Default::default() },
                ..Default::default()
            };
            let err = cfg.validate().unwrap_err();
            assert!(err.to_string().contains("SEMDOC_TEST_TOKEN"), "{err}");
            std::env::set_var("SEMDOC_TEST_TOKEN", "secret");
            cfg.validate().unwrap();
        });
    }

    #[test]
    fn validate_rejects_unset_api_key_env() {
        with_clean_env(|| {
            let raw = r#"
[rerank]
backend = "tei"
endpoint = "http://x:8000"
api_key_env = "SEMDOC_TEST_API_KEY"
"#;
            assert!(parse(raw).is_err());
            std::env::set_var("SEMDOC_TEST_API_KEY", "k");
            parse(raw).unwrap();
        });
    }

    #[test]
    fn validate_chunk_size_bounds() {
        with_clean_env(|| {
            let mk = |size: usize| DeploymentConfig {
                rerank: RerankConfig::None,
                chunk: ChunkConfig { size: Some(size), overlap: None },
                ..Default::default()
            };
            assert!(mk(63).validate().is_err());
            assert!(mk(8193).validate().is_err());
            mk(64).validate().unwrap();
            mk(8192).validate().unwrap();
        });
    }

    // --- env overrides ---

    #[test]
    fn env_full_http_embedding_config() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_EMBEDDER_BACKEND", "http");
            std::env::set_var("SEMDOC_EMBEDDER_HTTP_URL", "http://e:80/v1");
            std::env::set_var("SEMDOC_EMBEDDER_HTTP_MODEL", "bge-m3");
            std::env::set_var("SEMDOC_EMBEDDER_HTTP_TIMEOUT", "11");
            let mut cfg = DeploymentConfig::default();
            cfg.apply_env_overrides().unwrap();
            match &cfg.embedding {
                EmbeddingConfig::Http { url, model, timeout_secs, api_key_env } => {
                    assert_eq!(url, "http://e:80/v1");
                    assert_eq!(model, "bge-m3");
                    assert_eq!(*timeout_secs, 11);
                    assert!(api_key_env.is_none());
                }
                other => panic!("expected http embedder, got {other:?}"),
            }
        });
    }

    #[test]
    fn env_http_backend_requires_url_and_model() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_EMBEDDER_BACKEND", "http");
            let mut cfg = DeploymentConfig::default();
            assert!(cfg.apply_env_overrides().is_err());
            std::env::set_var("SEMDOC_EMBEDDER_HTTP_URL", "http://e:80/v1");
            let mut cfg = DeploymentConfig::default();
            assert!(cfg.apply_env_overrides().is_err());
        });
    }

    #[test]
    fn env_unknown_embedding_backend_is_error() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_EMBEDDER_BACKEND", "wat");
            let mut cfg = DeploymentConfig::default();
            assert!(cfg.apply_env_overrides().is_err());
        });
    }

    #[test]
    fn env_inline_api_key_routes_through_indirection() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_EMBEDDER_BACKEND", "http");
            std::env::set_var("SEMDOC_EMBEDDER_HTTP_URL", "http://e:80/v1");
            std::env::set_var("SEMDOC_EMBEDDER_HTTP_MODEL", "m");
            std::env::set_var("SEMDOC_EMBEDDER_HTTP_API_KEY", "key123");
            let mut cfg = DeploymentConfig::default();
            cfg.apply_env_overrides().unwrap();
            match &cfg.embedding {
                EmbeddingConfig::Http { api_key_env, .. } => {
                    assert_eq!(api_key_env.as_deref(), Some("SEMDOC_EMBEDDER_HTTP_API_KEY_INLINE"));
                }
                other => panic!("expected http embedder, got {other:?}"),
            }
        });
    }

    #[test]
    fn env_rerank_legacy_http_alias_maps_to_tei() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_RERANKER_BACKEND", "http");
            std::env::set_var("SEMDOC_RERANKER_HTTP_URL", "http://r:8000");
            let mut cfg = DeploymentConfig::default();
            cfg.apply_env_overrides().unwrap();
            match &cfg.rerank {
                RerankConfig::Tei { endpoint, .. } => assert_eq!(endpoint, "http://r:8000"),
                other => panic!("expected tei reranker, got {other:?}"),
            }
        });
    }

    #[test]
    fn env_rerank_requires_url() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_RERANKER_BACKEND", "tei");
            let mut cfg = DeploymentConfig::default();
            assert!(cfg.apply_env_overrides().is_err());
        });
    }

    #[test]
    fn env_chunk_and_server_and_tls() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_CHUNK_SIZE", "256");
            std::env::set_var("SEMDOC_CHUNK_OVERLAP", "32");
            std::env::set_var("SEMDOC_LISTEN", "0.0.0.0:9999");
            std::env::set_var("SEMDOC_TLS_INSECURE", "1");
            let mut cfg = DeploymentConfig { rerank: RerankConfig::None, ..Default::default() };
            cfg.apply_env_overrides().unwrap();
            cfg.validate().unwrap();
            assert_eq!(cfg.chunk.size, Some(256));
            assert_eq!(cfg.chunk.overlap, Some(32));
            assert_eq!(cfg.server.listen.as_deref(), Some("0.0.0.0:9999"));
            assert!(cfg.tls.insecure);
        });
    }

    #[test]
    fn env_bad_chunk_size_is_error() {
        with_clean_env(|| {
            std::env::set_var("SEMDOC_CHUNK_SIZE", "abc");
            let mut cfg = DeploymentConfig::default();
            assert!(cfg.apply_env_overrides().is_err());
        });
    }

    // --- file loading ---

    #[test]
    fn load_missing_explicit_file_is_error() {
        with_clean_env(|| {
            assert!(DeploymentConfig::load(Some("/nonexistent/Config.toml")).is_err());
        });
    }

    #[test]
    fn load_missing_default_file_uses_defaults_then_fails_validate() {
        // ./Config.toml in the crate root exists, so chdir into a temp dir.
        // std::env::set_current_dir is also process-global — reuse the lock.
        with_clean_env(|| {
            let tmp = std::env::temp_dir().join(format!("semdoc-cfg-test-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).unwrap();
            let prev = std::env::current_dir().unwrap();
            std::env::set_current_dir(&tmp).unwrap();
            let result = DeploymentConfig::load(None);
            std::env::set_current_dir(prev).unwrap();
            std::fs::remove_dir_all(&tmp).ok();
            // No file + no env → defaults; the default rerank endpoint is
            // empty, which validate() rejects. load() must surface that.
            let err = result.unwrap_err();
            assert!(err.to_string().contains("endpoint"), "{err}");
        });
    }

    #[test]
    fn load_via_semdoc_config_env() {
        with_clean_env(|| {
            let tmp = std::env::temp_dir().join(format!("semdoc-cfg-test2-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).unwrap();
            let path = tmp.join("my.toml");
            std::fs::write(
                &path,
                r#"
[rerank]
backend = "none"

[chunk]
size = 300
"#,
            )
            .unwrap();
            std::env::set_var("SEMDOC_CONFIG", path.to_str().unwrap());
            let cfg = DeploymentConfig::load(None).unwrap();
            assert_eq!(cfg.chunk.size, Some(300));
            assert!(matches!(cfg.rerank, RerankConfig::None));
            std::fs::remove_dir_all(&tmp).ok();
        });
    }

    // --- helpers ---

    #[test]
    fn server_token_reads_indirection() {
        with_clean_env(|| {
            let cfg = DeploymentConfig {
                rerank: RerankConfig::None,
                server: ServerConfig { token_env: Some("SEMDOC_TEST_TOKEN".into()), listen: None, ..Default::default() },
                ..Default::default()
            };
            std::env::set_var("SEMDOC_TEST_TOKEN", "tok");
            assert_eq!(cfg.server_token().as_deref(), Some("tok"));
        });
    }

    #[test]
    fn webhook_enabled_filters_events() {
        // no url → always off
        let off = ServerConfig::default();
        assert!(!off.webhook_enabled("add"));
        // url without explicit list → all events on
        let all = ServerConfig { webhook_url: Some("http://h/hook".into()), ..Default::default() };
        assert!(all.webhook_enabled("add"));
        assert!(all.webhook_enabled("delete"));
        assert!(all.webhook_enabled("update"));
        // explicit list → only listed events fire
        let filtered = ServerConfig {
            webhook_url: Some("http://h/hook".into()),
            webhook_events: Some(vec!["add".into()]),
            ..Default::default()
        };
        assert!(filtered.webhook_enabled("add"));
        assert!(!filtered.webhook_enabled("delete"));
        assert!(!filtered.webhook_enabled("update"));
    }

    #[test]
    fn webhook_secret_reads_env_indirection() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("SEMDOC_TEST_HOOK_SECRET");
        let cfg = ServerConfig {
            webhook_secret_env: Some("SEMDOC_TEST_HOOK_SECRET".into()),
            ..Default::default()
        };
        // unset → None
        assert!(cfg.webhook_secret().is_none());
        // empty string counts as unset
        std::env::set_var("SEMDOC_TEST_HOOK_SECRET", "");
        assert!(cfg.webhook_secret().is_none());
        // set → value
        std::env::set_var("SEMDOC_TEST_HOOK_SECRET", "s3cret");
        assert_eq!(cfg.webhook_secret().as_deref(), Some("s3cret"));
        std::env::remove_var("SEMDOC_TEST_HOOK_SECRET");
    }

    #[test]
    fn tls_apply_env_exports_flags() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("SEMDOC_TLS_INSECURE");
        std::env::remove_var("SEMDOC_CA_BUNDLE");
        // defaults: nothing exported
        TlsConfig::default().apply_env();
        assert!(std::env::var("SEMDOC_TLS_INSECURE").is_err());
        assert!(std::env::var("SEMDOC_CA_BUNDLE").is_err());
        // insecure + ca_bundle → both exported
        let cfg = TlsConfig {
            insecure: true,
            ca_bundle: Some("/etc/semdoc/ca.pem".into()),
        };
        cfg.apply_env();
        assert_eq!(std::env::var("SEMDOC_TLS_INSECURE").as_deref(), Ok("1"));
        assert_eq!(
            std::env::var("SEMDOC_CA_BUNDLE").as_deref(),
            Ok("/etc/semdoc/ca.pem")
        );
        std::env::remove_var("SEMDOC_TLS_INSECURE");
        std::env::remove_var("SEMDOC_CA_BUNDLE");
    }

    #[test]
    fn apply_chunk_env_pushes_values() {
        with_clean_env(|| {
            let cfg = DeploymentConfig {
                rerank: RerankConfig::None,
                chunk: ChunkConfig { size: Some(700), overlap: Some(70) },
                ..Default::default()
            };
            cfg.apply_chunk_env();
            assert_eq!(std::env::var("SEMDOC_CHUNK_SIZE").unwrap(), "700");
            assert_eq!(std::env::var("SEMDOC_CHUNK_OVERLAP").unwrap(), "70");
        });
    }
}
