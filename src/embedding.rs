// SPDX-License-Identifier: MIT OR Apache-2.0
//! Embedding backends: in-process ONNX (feature `onnx`) and HTTP
//! (OpenAI-compatible `/v1/embeddings`).
//!
//! The embedder is constructed from env vars (feature parity with semrag):
//! - `SEMDOC_EMBEDDER_BACKEND` = `onnx` (default) | `http`
//! - onnx: model dir via `SEMDOC_EMBEDDER_DIR` (default
//!   `~/.cache/semdoc/models/bge-m3`)
//! - http: `SEMDOC_EMBEDDER_HTTP_URL` (base incl. `/v1`),
//!   `SEMDOC_EMBEDDER_HTTP_MODEL`, optional `SEMDOC_EMBEDDER_HTTP_API_KEY`

use anyhow::Result;

/// Load the embedder from the environment (legacy path, kept for backward
/// compatibility). Prefer `load_from_config` via the deployment config.
pub fn load_from_env() -> Result<Embedder> {
    Embedder::load(&crate::config::DeploymentConfig::load(None)?.embedding)
}

impl Embedder {
    /// Load from the deployment config's [embedding] section.
    pub fn load(cfg: &crate::config::EmbeddingConfig) -> Result<Embedder> {
        match cfg {
            crate::config::EmbeddingConfig::Onnx { dir } => {
                #[cfg(feature = "onnx")]
                {
                    let resolved = match dir {
                        Some(p) => std::path::PathBuf::from(p.trim()),
                        None => dirs_home()
                            .map(|h| h.join(".cache/semdoc/models/bge-m3"))
                            .ok_or_else(|| anyhow::anyhow!("cannot resolve home dir; set embedding.dir"))?,
                    };
                    OnnxEmbedder::load(resolved.clone()).map_err(|e| {
                        anyhow::anyhow!(
                            "{e:#}\nhint: no deployment config found. Either create semdoc.config.toml \
                             with [embedding], set SEMDOC_EMBEDDER_* env vars, or place the ONNX model at {}",
                            resolved.display()
                        )
                    }).map(Embedder::Onnx)
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = dir;
                    Err(anyhow::anyhow!(
                        "embedding backend onnx requires the `onnx` feature \
                         (build with: cargo build --release --bins --features onnx)"
                    ))
                }
            }
            crate::config::EmbeddingConfig::Http { url, model, api_key_env, .. } => {
                let api_key = api_key_env
                    .as_ref()
                    .and_then(|e| std::env::var(e).ok());
                Ok(Embedder::Http(HttpEmbedder::new(url.trim(), model.trim(), api_key)))
            }
        }
    }
}

/// Probe the embedding dimension of the configured backend without loading
/// the full model where possible. For http backends this encodes a probe
/// string; for onnx it loads the model (dim is part of the output shape).
pub async fn probe_dim_from_env() -> Result<usize> {
    let e = load_from_env()?;
    let v = e.encode_blocking("dimension probe".to_string()).await?;
    Ok(v.len())
}

pub fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Embedding backend. `encode_blocking` runs the (potentially CPU-bound or
/// sync-I/O) encode on tokio's blocking pool.
pub enum Embedder {
    #[cfg(feature = "onnx")]
    Onnx(OnnxEmbedder),
    Http(HttpEmbedder),
}

impl Clone for Embedder {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(o) => Self::Onnx(o.clone()),
            Self::Http(h) => Self::Http(h.clone()),
        }
    }
}

impl Embedder {
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            #[cfg(feature = "onnx")]
            Self::Onnx(o) => o.encode(text),
            Self::Http(h) => h.encode(text),
        }
    }

    pub async fn encode_blocking(&self, text: String) -> Result<Vec<f32>> {
        let this = self.clone();
        tokio::task::spawn_blocking(move || this.encode(&text))
            .await
            .map_err(|e| anyhow::anyhow!("embedder spawn_blocking join: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// ONNX backend
// ---------------------------------------------------------------------------

#[cfg(feature = "onnx")]
pub mod onnx_impl {
    use anyhow::Result;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::TensorRef;
    use tokenizers::Tokenizer;

    /// bge-m3 ONNX. Pre-pooled `sentence_embedding` output (Float32 [1, dim]),
    /// no mean-pooling needed. Inputs: `input_ids` + `attention_mask`
    /// (Int64 [1, L]); no token_type_ids (XLM-R has no segment ids).
    pub struct OnnxEmbedder {
        inner: std::sync::Arc<std::sync::Mutex<OnnxInner>>,
    }

    struct OnnxInner {
        session: Session,
        tokenizer: Tokenizer,
    }

    impl Clone for OnnxEmbedder {
        fn clone(&self) -> Self {
            Self { inner: std::sync::Arc::clone(&self.inner) }
        }
    }

    impl OnnxEmbedder {
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
            Ok(Self { inner: std::sync::Arc::new(std::sync::Mutex::new(OnnxInner { session, tokenizer })) })
        }

        pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
            let mut inner = self.inner.lock().expect("embedder mutex poisoned");

            // Truncation strategy is unset by from_file(); configure per-call
            // (cheap relative to the forward pass) — same as semrag.
            let mut tokenizer = inner.tokenizer.clone();
            tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                max_length: 8192,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| anyhow::anyhow!("with_truncation: {e:?}"))?;

            let enc = tokenizer.encode(text, true)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
            let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
            let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&v| v as i64).collect();
            let seq_len = ids.len();
            if seq_len == 0 {
                anyhow::bail!("empty text cannot be embedded");
            }

            let ids_arr = ndarray::Array2::from_shape_vec((1, seq_len), ids)
                .map_err(|e| anyhow::anyhow!("ids shape: {e}"))?;
            let mask_arr = ndarray::Array2::from_shape_vec((1, seq_len), mask)
                .map_err(|e| anyhow::anyhow!("mask shape: {e}"))?;

            let ids_view = TensorRef::from_array_view(ids_arr.view())
                .map_err(|e| anyhow::anyhow!("from_array_view(ids): {e}"))?;
            let mask_view = TensorRef::from_array_view(mask_arr.view())
                .map_err(|e| anyhow::anyhow!("from_array_view(mask): {e}"))?;

            let outputs = inner.session.run(
                ort::inputs!["input_ids" => ids_view, "attention_mask" => mask_view]
            )
            .map_err(|e| anyhow::anyhow!("session.run: {e}"))?;

            let arr = outputs["sentence_embedding"]
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("try_extract_array(sentence_embedding): {e}"))?;
            let arr = arr.into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| anyhow::anyhow!("into_dimensionality: {e}"))?;
            let row = arr.outer_iter().next()
                .ok_or_else(|| anyhow::anyhow!("sentence_embedding had zero rows"))?;
            Ok(row.iter().copied().collect())
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_impl::OnnxEmbedder;

// ---------------------------------------------------------------------------
// HTTP backend
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct HttpEmbedder {
    http: reqwest::blocking::Client,
    url: String,
    model: String,
    api_key: Option<String>,
}

impl HttpEmbedder {
    /// `url` is the base URL including `/v1`; POSTs to `{url}/embeddings`.
    pub fn new(url: &str, model: &str, api_key: Option<String>) -> Self {
        let http = reqwest::blocking::Client::builder()
            .danger_accept_invalid_certs(
                std::env::var("SEMDOC_TLS_INSECURE").is_ok_and(|v| v == "1" || v == "true"),
            )
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(60))
            .pool_idle_timeout(std::time::Duration::from_secs(90))
            .pool_max_idle_per_host(64)
            .build()
            .expect("http client build");
        Self { http, url: url.to_string(), model: model.to_string(), api_key }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.url.trim_end_matches('/'));
        let body = serde_json::json!({ "model": self.model, "input": text });
        let mut req = self.http.post(&url);
        if let Some(ref key) = self.api_key {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")) {
                req = req.header("Authorization", v);
            }
        }
        let resp = req.json(&body).send()
            .map_err(|e| anyhow::anyhow!("embed POST {url}: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let t = resp.text().unwrap_or_default();
            anyhow::bail!("embed HTTP {status}: {t}");
        }
        let parsed: OpenAiEmbeddingResp = resp.json()
            .map_err(|e| anyhow::anyhow!("embed JSON decode: {e}"))?;
        parsed.data.into_iter().next().map(|i| i.embedding)
            .ok_or_else(|| anyhow::anyhow!("embeddings response had no data[0]"))
    }
}

#[derive(serde::Deserialize)]
struct OpenAiEmbeddingResp {
    data: Vec<OpenAiEmbeddingItem>,
}

#[derive(serde::Deserialize)]
struct OpenAiEmbeddingItem {
    embedding: Vec<f32>,
}
