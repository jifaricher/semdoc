// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rerank backends: in-process ONNX cross-encoder (feature `onnx`),
//! TEI-compatible HTTP `/rerank`, and OpenAI-compatible `/v1/rerank`.
//!
//! HTTP variants retry transient failures (network errors / 5xx) up to 2
//! extra times with 250/750ms backoff; 4xx fails immediately. Callers
//! additionally degrade to ANN order when rerank ultimately fails, so a
//! dead reranker service never fails a query.

use anyhow::Result;
use std::time::Duration;

pub enum Reranker {
    #[cfg(feature = "onnx")]
    Onnx(Box<OnnxReranker>),
    Http(HttpReranker),
    OpenAi(OpenAiReranker),
}

impl Reranker {
    pub fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>> {
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(o) => o.rerank(query, docs),
            Self::Http(h) => h.rerank(query, docs),
            Self::OpenAi(o) => o.rerank(query, docs),
        }
    }
}

/// Load from the environment (legacy path, kept for backward compatibility).
/// Prefer `load_from_config` via the deployment config.
pub fn load_from_env() -> Result<Reranker> {
    Reranker::load(&crate::config::DeploymentConfig::load(None)?.rerank)
}

impl Reranker {
    /// Load from the deployment config's [rerank] section. `None` variant →
    /// Err (callers treat that as "no reranker", degrading queries).
    pub fn load(cfg: &crate::config::RerankConfig) -> Result<Reranker> {
        match cfg {
            crate::config::RerankConfig::None => Err(anyhow::anyhow!("rerank disabled by config")),
            crate::config::RerankConfig::Onnx { dir } => {
                #[cfg(feature = "onnx")]
                {
                    let resolved = match dir {
                        Some(p) => std::path::PathBuf::from(p.trim()),
                        None => crate::embedding::dirs_home()
                            .map(|h| h.join(".cache/semdoc/models/bge-reranker-v2-m3"))
                            .ok_or_else(|| anyhow::anyhow!("cannot resolve home; set rerank.dir"))?,
                    };
                    OnnxReranker::load(resolved).map(|o| Reranker::Onnx(Box::new(o)))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = dir;
                    Err(anyhow::anyhow!("rerank backend onnx requires the `onnx` feature"))
                }
            }
            crate::config::RerankConfig::Tei { endpoint, api_key_env, timeout_secs: _ } => {
                let api_key = api_key_env
                    .as_ref()
                    .and_then(|e| std::env::var(e).ok());
                Ok(Reranker::Http(HttpReranker::new(endpoint.trim(), api_key)))
            }
            crate::config::RerankConfig::Openai { endpoint, model, api_key_env, timeout_secs: _ } => {
                let api_key = api_key_env
                    .as_ref()
                    .and_then(|e| std::env::var(e).ok());
                Ok(Reranker::OpenAi(OpenAiReranker::new(endpoint.trim(), model.clone(), api_key)))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX backend
// ---------------------------------------------------------------------------

#[cfg(feature = "onnx")]
pub mod onnx_impl {
    use super::*;
    use ndarray::{Array2, Ix2};
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::TensorRef;
    use tokenizers::Tokenizer;

    pub struct OnnxReranker {
        inner: std::sync::Mutex<OnnxInner>,
    }

    struct OnnxInner {
        session: Session,
        tokenizer: Tokenizer,
    }

    impl OnnxReranker {
        pub fn load(model_dir: std::path::PathBuf) -> Result<Self> {
            let onnx_path = model_dir.join("model.onnx");
            let tok_path = model_dir.join("tokenizer.json");
            let session = Session::builder()
                .map_err(|e| anyhow::anyhow!("Session::builder: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow::anyhow!("with_optimization_level: {e}"))?
                .with_intra_threads(4)
                .map_err(|e| anyhow::anyhow!("with_intra_threads: {e}"))?
                .commit_from_file(&onnx_path)
                .map_err(|e| anyhow::anyhow!("commit_from_file({}): {e}", onnx_path.display()))?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow::anyhow!("Tokenizer::from_file({}): {e}", tok_path.display()))?;
            Ok(Self { inner: std::sync::Mutex::new(OnnxInner { session, tokenizer }) })
        }

        pub fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>> {
            let mut inner = self.inner.lock().expect("reranker mutex poisoned");
            let pairs: Vec<(String, String)> = docs
                .iter()
                .map(|d| (query.to_string(), d.clone()))
                .collect();

            // bge-reranker-v2-m3 (XLM-R) 8192-token context; truncation unset
            // by from_file() — set per call. Pad token id 1 (XLM-R config).
            let mut tokenizer = inner.tokenizer.clone();
            tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                max_length: 8192,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| anyhow::anyhow!("with_truncation: {e:?}"))?;

            let encs = tokenizer.encode_batch(pairs, true)
                .map_err(|e| anyhow::anyhow!("encode_batch: {e}"))?;
            let max_len = encs.iter().map(|e| e.len()).max().unwrap_or(0);
            if max_len == 0 {
                anyhow::bail!("no documents to rerank");
            }
            let b = docs.len();
            let pad_id: i64 = 1;

            let mut ids = Vec::with_capacity(b * max_len);
            let mut mask = Vec::with_capacity(b * max_len);
            for e in &encs {
                let n = e.len();
                let ids_row = e.get_ids();
                let mask_row = e.get_attention_mask();
                for j in 0..max_len {
                    if j < n {
                        ids.push(ids_row[j] as i64);
                        mask.push(mask_row[j] as i64);
                    } else {
                        ids.push(pad_id);
                        mask.push(0);
                    }
                }
            }

            let ids_arr = Array2::from_shape_vec((b, max_len), ids)
                .map_err(|e| anyhow::anyhow!("ids shape: {e}"))?;
            let mask_arr = Array2::from_shape_vec((b, max_len), mask)
                .map_err(|e| anyhow::anyhow!("mask shape: {e}"))?;
            let ids_view = TensorRef::from_array_view(ids_arr.view())
                .map_err(|e| anyhow::anyhow!("from_array_view(ids): {e}"))?;
            let mask_view = TensorRef::from_array_view(mask_arr.view())
                .map_err(|e| anyhow::anyhow!("from_array_view(mask): {e}"))?;

            let outputs = inner.session.run(
                ort::inputs!["input_ids" => ids_view, "attention_mask" => mask_view]
            )
            .map_err(|e| anyhow::anyhow!("session.run: {e}"))?;
            let arr = outputs["logits"]
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("try_extract_array(logits): {e}"))?;
            let arr = arr.into_dimensionality::<Ix2>()
                .map_err(|e| anyhow::anyhow!("into_dimensionality: {e}"))?;

            let mut indexed: Vec<(usize, f32)> = arr
                .outer_iter()
                .map(|row| row.iter().next().copied().unwrap_or(f32::NAN))
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            Ok(indexed)
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_impl::OnnxReranker;

// ---------------------------------------------------------------------------
// HTTP backends
// ---------------------------------------------------------------------------

fn http_client(timeout_secs: u64) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .danger_accept_invalid_certs(
            std::env::var("SEMDOC_TLS_INSECURE").is_ok_and(|v| v == "1" || v == "true"),
        )
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(timeout_secs))
        .pool_idle_timeout(Duration::from_secs(0))
        .build()
        .expect("reranker http client build")
}

/// Retry policy shared by both HTTP backends: 2 extra attempts on transient
/// failures with 250/750ms backoff. Transient = network-layer errors
/// (message starts with "rerank POST") or 5xx statuses; 4xx is deterministic.
fn with_transient_retry<F>(mut f: F) -> Result<Vec<(usize, f32)>>
where
    F: FnMut() -> Result<Vec<(usize, f32)>>,
{
    let mut attempt = 0;
    loop {
        match f() {
            Ok(r) => return Ok(r),
            Err(e) if attempt < 2 && is_transient(&e) => {
                attempt += 1;
                let backoff = Duration::from_millis(250 * 3u64.pow(attempt - 1));
                eprintln!(
                    "[reranker] transient failure (attempt {attempt}/2), retrying in {backoff:?}: {e:#}"
                );
                std::thread::sleep(backoff);
            }
            Err(e) => return Err(e),
        }
    }
}

fn is_transient(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}");
    s.starts_with("rerank POST") || s.starts_with("rerank HTTP 5")
}

pub struct HttpReranker {
    http: reqwest::blocking::Client,
    url: String,
    api_key: Option<String>,
}

impl HttpReranker {
    /// `url` is the TEI base; `/rerank` is appended per call.
    pub fn new(url: &str, api_key: Option<String>) -> Self {
        Self { http: http_client(120), url: url.to_string(), api_key }
    }

    pub fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>> {
        with_transient_retry(|| self.rerank_once(query, docs))
    }

    fn rerank_once(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>> {
        let url = format!("{}/rerank", self.url.trim_end_matches('/'));
        let body = serde_json::json!({
            "query": query,
            "texts": docs,
            "return_text": false,
        });
        let mut req = self.http.post(&url);
        if let Some(ref key) = self.api_key {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")) {
                req = req.header("Authorization", v);
            }
        }
        let resp = req.json(&body).send().map_err(|e| {
            anyhow::anyhow!("rerank POST {url}: {e}")
        })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().unwrap_or_default();
            anyhow::bail!("rerank HTTP {status}: {t}");
        }
        let items: Vec<TeiRerankItem> = resp.json()
            .map_err(|e| anyhow::anyhow!("rerank JSON decode: {e}"))?;
        let mut indexed: Vec<(usize, f32)> = items.into_iter().map(|i| (i.index, i.score)).collect();
        // TEI returns score-desc but re-sort defensively (Jina/Infinity/etc
        // share the shape without the ordering guarantee).
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed)
    }
}

#[derive(serde::Deserialize)]
struct TeiRerankItem {
    index: usize,
    score: f32,
}

pub struct OpenAiReranker {
    http: reqwest::blocking::Client,
    url: String,
    model: Option<String>,
    api_key: Option<String>,
}

impl OpenAiReranker {
    /// `url` base must include `/v1`; `/rerank` is appended per call.
    pub fn new(url: &str, model: Option<String>, api_key: Option<String>) -> Self {
        Self { http: http_client(120), url: url.to_string(), model, api_key }
    }

    pub fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>> {
        with_transient_retry(|| self.rerank_once(query, docs))
    }

    fn rerank_once(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>> {
        let url = format!("{}/rerank", self.url.trim_end_matches('/'));
        let mut body = serde_json::json!({
            "query": query,
            "documents": docs,
        });
        if let Some(model) = &self.model {
            body["model"] = serde_json::Value::String(model.clone());
        }
        let mut req = self.http.post(&url);
        if let Some(ref key) = self.api_key {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")) {
                req = req.header("Authorization", v);
            }
        }
        let resp = req.json(&body).send().map_err(|e| anyhow::anyhow!("rerank POST {url}: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().unwrap_or_default();
            anyhow::bail!("rerank HTTP {status}: {t}");
        }
        let parsed: OpenAiRerankResp = resp.json()
            .map_err(|e| anyhow::anyhow!("rerank JSON decode: {e}"))?;
        let mut indexed: Vec<(usize, f32)> = parsed
            .results
            .into_iter()
            .map(|i| (i.index, i.relevance_score))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed)
    }
}

/// OpenAI-compatible `/v1/rerank` response. `usage` optional (lenient).
#[derive(serde::Deserialize)]
struct OpenAiRerankResp {
    results: Vec<OpenAiRerankItem>,
}

#[derive(serde::Deserialize)]
struct OpenAiRerankItem {
    index: usize,
    relevance_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(msg: &str) -> anyhow::Error {
        anyhow::anyhow!("{msg}")
    }

    #[test]
    fn transient_detection() {
        assert!(is_transient(&err("rerank POST http://x: connection refused")));
        assert!(is_transient(&err("rerank HTTP 500 Internal Server Error: boom")));
        assert!(is_transient(&err("rerank HTTP 503 Service Unavailable")));
        // 4xx is deterministic — not retried.
        assert!(!is_transient(&err("rerank HTTP 401 Unauthorized")));
        assert!(!is_transient(&err("rerank HTTP 404 Not Found")));
        assert!(!is_transient(&err("rerank JSON decode: bad shape")));
    }

    #[test]
    fn retry_returns_first_success_without_sleeping() {
        let mut calls = 0;
        let out = with_transient_retry(|| {
            calls += 1;
            Ok(vec![(1usize, 0.5f32)])
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(out, vec![(1, 0.5)]);
    }

    #[test]
    fn retry_succeeds_after_transient_failures() {
        let mut calls = 0;
        let out = with_transient_retry(|| {
            calls += 1;
            if calls < 3 {
                Err(err("rerank HTTP 503"))
            } else {
                Ok(vec![(0, 1.0)])
            }
        })
        .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(out, vec![(0, 1.0)]);
    }

    #[test]
    fn retry_gives_up_after_two_extra_attempts() {
        let mut calls = 0;
        let res = with_transient_retry(|| {
            calls += 1;
            Err::<Vec<(usize, f32)>, _>(err("rerank HTTP 500"))
        });
        assert!(res.is_err());
        assert_eq!(calls, 3, "1 initial + 2 retries");
    }

    #[test]
    fn non_transient_error_fails_immediately() {
        let mut calls = 0;
        let res = with_transient_retry(|| {
            calls += 1;
            Err::<Vec<(usize, f32)>, _>(err("rerank HTTP 401 Unauthorized"))
        });
        assert!(res.is_err());
        assert_eq!(calls, 1);
    }
}
