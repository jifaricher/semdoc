// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared TLS configuration for all outbound HTTP clients (embedder,
//! reranker, graph plugin). One place decides: verify against the system
//! store, add a company CA bundle, or (explicitly, for dev) skip validation.
//!
//! Resolution order (highest wins):
//!   1. `SEMDOC_TLS_INSECURE` / `SEMDOC_CA_BUNDLE` env vars
//!      (deployment config applies `[tls]` via `TlsConfig::apply_env`, so
//!      Config.toml values land here too; an operator-exported env var
//!      simply matches what apply_env would set)
//!   2. defaults: verify with the system certificate store

/// Env var name for the insecure flag (kept for backward compatibility).
pub const ENV_INSECURE: &str = "SEMDOC_TLS_INSECURE";
/// Env var name for a PEM CA bundle to trust in addition to the system store.
pub const ENV_CA_BUNDLE: &str = "SEMDOC_CA_BUNDLE";

pub fn insecure_enabled() -> bool {
    std::env::var(ENV_INSECURE).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

pub fn ca_bundle_path() -> Option<String> {
    std::env::var(ENV_CA_BUNDLE).ok().filter(|s| !s.is_empty())
}

macro_rules! load_ca_bundle {
    ($builder:expr) => {{
        let mut builder = $builder;
        match ca_bundle_path() {
            None => {}
            Some(path) => match std::fs::read(&path) {
                // tls_certs_merge: roots are merged with the platform
                // verifier's store (0.13's recommended API; plain
                // add_root_certificate no longer feeds the platform
                // verifier on this version).
                Ok(pem) => match reqwest::Certificate::from_pem_bundle(&pem) {
                    Ok(certs) => builder = builder.tls_certs_merge(certs),
                    Err(e) => {
                        eprintln!("[tls] CA bundle {path}: parse error ({e}); using system store only");
                    }
                },
                Err(e) => {
                    eprintln!("[tls] CA bundle {path}: unreadable ({e}); using system store only");
                }
            },
        }
        builder
    }};
}

/// Configure a reqwest async-client builder with the process TLS policy.
/// All HTTP clients must go through this so behavior stays consistent.
pub fn apply_async(builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    let builder = load_ca_bundle!(builder);
    if insecure_enabled() {
        eprintln!(
            "WARNING: {ENV_INSECURE} is set — TLS certificate validation is \
             disabled; Bearer tokens can be intercepted"
        );
        builder.danger_accept_invalid_certs(true)
    } else {
        builder
    }
}

/// Same for the blocking client builder used by the embedder and rerankers.
pub fn apply_blocking(builder: reqwest::blocking::ClientBuilder) -> reqwest::blocking::ClientBuilder {
    let builder = load_ca_bundle!(builder);
    if insecure_enabled() {
        eprintln!(
            "WARNING: {ENV_INSECURE} is set — TLS certificate validation is \
             disabled; Bearer tokens can be intercepted"
        );
        builder.danger_accept_invalid_certs(true)
    } else {
        builder
    }
}

/// Build a blocking reqwest client on a plain OS thread.
///
/// reqwest 0.13's blocking client construction creates and drops a temporary
/// tokio runtime inline; doing that within an async context panics ("Cannot
/// drop a runtime in a context where blocking is not allowed"). Every call
/// site (CLI add/query, MCP server, semdoc-server startup) runs inside a
/// tokio runtime, so the build must happen off it.
pub fn build_blocking(
    builder: reqwest::blocking::ClientBuilder,
) -> anyhow::Result<reqwest::blocking::Client> {
    std::thread::spawn(move || builder.build())
        .join()
        .map_err(|_| anyhow::anyhow!("blocking http client build thread panicked"))?
        .map_err(|e| anyhow::anyhow!("blocking http client build: {e}"))
}
