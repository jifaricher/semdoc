// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph-reasoning plugin layer.
//!
//! [`GraphPlugin`] abstracts multi-hop retrieval backends. Implementations:
//! - [`NoGraph`] — graph disabled; every query degrades to semantic/None
//! - [`LightragServer`] — LightRAG server over HTTP (zero Python)
//! - `LightragEmbedded` — PyO3 in-process (feature `lightrag-embedded`)

use anyhow::Result;
use async_trait::async_trait;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphMode {
    /// Hybrid keyword+vector retrieval with LLM synthesis.
    Hybrid,
    /// Structured data query (entities/relationships/chunks), no LLM answer.
    Data,
}

/// Graph-layer context sizing knobs, passed through to lightrag's QueryParam.
/// Missing values keep lightrag defaults (chunk_top_k=20, entity=6000,
/// relation=8000, total=30000 tokens).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphQueryParams {
    pub chunk_top_k: Option<usize>,
    pub max_entity_tokens: Option<usize>,
    pub max_relation_tokens: Option<usize>,
    pub max_total_tokens: Option<usize>,
}

impl GraphQueryParams {
    fn to_body(self) -> serde_json::Value {
        let mut b = serde_json::Map::new();
        if let Some(v) = self.chunk_top_k {
            b.insert("chunk_top_k".into(), v.into());
        }
        if let Some(v) = self.max_entity_tokens {
            b.insert("max_entity_tokens".into(), v.into());
        }
        if let Some(v) = self.max_relation_tokens {
            b.insert("max_relation_tokens".into(), v.into());
        }
        if let Some(v) = self.max_total_tokens {
            b.insert("max_total_tokens".into(), v.into());
        }
        serde_json::Value::Object(b)
    }
}

#[derive(Debug, Clone)]
pub struct GraphAnswer {
    /// LLM-synthesized answer (hybrid) or JSON-encoded structured result
    /// (data mode). Callers treat it as opaque content.
    pub content: String,
    /// True when the answer came from the fallback (no real graph backend).
    pub degraded: bool,
}

/// A graph backend. `health()` is checked at startup and before queries —
/// failures degrade, they never crash the server.
#[async_trait]
pub trait GraphPlugin: Send + Sync {
    fn name(&self) -> &str;
    async fn health(&self) -> Result<()>;
    async fn insert(&self, docs: Vec<(String, String)>) -> Result<()>;
    async fn delete(&self, doc_id: &str) -> Result<()>;
    async fn query(
        &self,
        text: &str,
        mode: GraphMode,
        limit: usize,
        params: &GraphQueryParams,
    ) -> Result<GraphAnswer>;
}

/// Graph disabled: queries return a degraded marker (or are never routed
/// here), inserts are no-ops.
pub struct NoGraph;

#[async_trait]
impl GraphPlugin for NoGraph {
    fn name(&self) -> &str {
        "none"
    }
    async fn health(&self) -> Result<()> {
        Err(anyhow::anyhow!("graph plugin disabled"))
    }
    async fn insert(&self, _docs: Vec<(String, String)>) -> Result<()> {
        Ok(())
    }
    async fn delete(&self, _doc_id: &str) -> Result<()> {
        Ok(())
    }
    async fn query(
        &self,
        _text: &str,
        _mode: GraphMode,
        _limit: usize,
        _params: &GraphQueryParams,
    ) -> Result<GraphAnswer> {
        Err(anyhow::anyhow!("graph plugin disabled"))
    }
}

// ---------------------------------------------------------------------------
// LightRAG server (HTTP)
// ---------------------------------------------------------------------------

pub struct LightragServer {
    endpoint: String,
    api_key: Option<String>,
    http: reqwest::Client,
    insert_semaphore: tokio::sync::Semaphore,
    insert_timeout_secs: u64,
    delete_timeout_secs: u64,
}

impl LightragServer {
    pub fn new(
        endpoint: &str,
        api_key: Option<String>,
        insert_concurrency: usize,
        insert_timeout_secs: u64,
        delete_timeout_secs: u64,
    ) -> Self {
        let http = crate::tls::apply_async(reqwest::Client::builder())
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("lightrag http client");
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key,
            http,
            insert_semaphore: tokio::sync::Semaphore::new(insert_concurrency.max(1)),
            insert_timeout_secs,
            delete_timeout_secs,
        }
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(k) => req.bearer_auth(k),
            None => req,
        }
    }
}

#[async_trait]
impl GraphPlugin for LightragServer {
    fn name(&self) -> &str {
        "lightrag-server"
    }

    async fn health(&self) -> Result<()> {
        let url = format!("{}/health", self.endpoint);
        let resp = self
            .auth(self.http.get(&url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("lightrag health {url}: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("lightrag health: HTTP {}", resp.status()))
        }
    }

    async fn insert(&self, docs: Vec<(String, String)>) -> Result<()> {
        for (text, doc_id) in docs {
            let _permit = self.insert_semaphore.acquire().await?;
            // lightrag >=1.5 requires `file_source` (it's the document's
            // identity for dedup/409 on re-insert); `ids` is not accepted on
            // this route. We pass the semdoc doc id as the file_source so
            // deletes can target the same id.
            let url = format!("{}/documents/text", self.endpoint);
            let body = serde_json::json!({
                "text": text,
                "file_source": format!("{doc_id}.txt"),
            });
            let resp = self
                .auth(self.http.post(&url))
                .json(&body)
                .timeout(std::time::Duration::from_secs(self.insert_timeout_secs))
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("lightrag insert {doc_id}: {e}"))?;
            if !resp.status().is_success() {
                let status = resp.status();
                let t = resp.text().await.unwrap_or_default();
                // Same file_source already tracked: lightrag refuses with 409
                // until the old doc is deleted. Treat as an error — callers
                // delete-then-insert to replace.
                anyhow::bail!("lightrag insert {doc_id}: HTTP {status}: {t}");
            }
        }
        Ok(())
    }

    async fn delete(&self, doc_id: &str) -> Result<()> {
        // lightrag derives a document's internal id as
        // md5(file_source) = md5("<docid>.txt") when the doc was inserted
        // with `file_source` (see lightrag pipeline.py known_source branch).
        // Deleting by the raw semdoc id fails with "Document not found".
        // Derive the same internal id here; fall back to the raw id for docs
        // that were inserted by other means.
        use md5::Digest;
        let digest = format!("{:x}", md5::Md5::new_with_prefix(doc_id).chain_update(".txt").finalize());
        let url = format!("{}/documents/delete_document", self.endpoint);
        let ids = serde_json::json!([format!("doc-{digest}"), doc_id]);
        let resp = self
            .auth(self.http.delete(&url))
            .json(&serde_json::json!({ "doc_ids": ids }))
            .timeout(std::time::Duration::from_secs(self.delete_timeout_secs))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("lightrag delete {doc_id}: {e}"))?;
        if !resp.status().is_success() {
            anyhow::bail!("lightrag delete {doc_id}: HTTP {}", resp.status());
        }
        // {"status":"busy"} is a 200 — the destructive slot was taken; the
        // doc stays. Surface that to the caller rather than silently dropping.
        let body: serde_json::Value = resp.json().await.unwrap_or_default();
        if body["status"] == "busy" || body["status"] == "not_allowed" {
            anyhow::bail!(
                "lightrag delete {doc_id}: {} — {}",
                body["status"].as_str().unwrap_or("busy"),
                body["message"].as_str().unwrap_or("pipeline busy")
            );
        }
        Ok(())
    }

    async fn query(
        &self,
        text: &str,
        mode: GraphMode,
        limit: usize,
        params: &GraphQueryParams,
    ) -> Result<GraphAnswer> {
        // GraphMode::Data returns the raw structured JSON body (status/data
        // envelope from lightrag's /query/data); the caller (server handler)
        // maps data.chunks[].doc_id back to local documents. Hybrid returns
        // the LLM-synthesized answer text.
        let (url, mut body, timeout) = match mode {
            GraphMode::Hybrid => (
                format!("{}/query", self.endpoint),
                serde_json::json!({ "query": text, "mode": "hybrid", "top_k": limit }),
                120,
            ),
            GraphMode::Data => (
                format!("{}/query/data", self.endpoint),
                serde_json::json!({ "query": text, "mode": "hybrid", "top_k": limit }),
                60,
            ),
        };
        if let (serde_json::Value::Object(base), serde_json::Value::Object(extra)) =
            (&mut body, params.to_body())
        {
            base.extend(extra);
        }
        let resp = self
            .auth(self.http.post(&url))
            .json(&body)
            .timeout(std::time::Duration::from_secs(timeout))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("lightrag query: {e}"))?;
        if !resp.status().is_success() {
            anyhow::bail!("lightrag query: HTTP {}", resp.status());
        }
        let content = resp.text().await?;
        Ok(GraphAnswer { content, degraded: false })
    }
}

// ---------------------------------------------------------------------------
// Plugin construction
// ---------------------------------------------------------------------------

pub async fn build_graph_plugin(
    config: &Option<crate::schema::GraphPluginConfig>,
) -> Result<Box<dyn GraphPlugin>> {
    match config {
        None => Ok(Box::new(NoGraph)),
        Some(crate::schema::GraphPluginConfig::LightragServer {
            endpoint,
            api_key_env,
            insert_concurrency,
            insert_timeout_secs,
            delete_timeout_secs,
            ..
        }) => {
            let api_key = api_key_env
                .as_ref()
                .and_then(|e| std::env::var(e).ok());
            let plugin = LightragServer::new(
                endpoint,
                api_key,
                *insert_concurrency,
                *insert_timeout_secs,
                *delete_timeout_secs,
            );
            if let Err(e) = plugin.health().await {
                eprintln!("[graph] lightrag-server unhealthy at startup: {e:#}");
                eprintln!("[graph] queries will degrade to semantic search until it recovers");
            }
            Ok(Box::new(plugin))
        }
        Some(crate::schema::GraphPluginConfig::LightragEmbedded { .. }) => {
            #[cfg(feature = "lightrag-embedded")]
            {
                Err(anyhow::anyhow!(
                    "lightrag-embedded backend is a build-time stub in this version; \
                     use backend = \"lightrag-server\""
                ))
            }
            #[cfg(not(feature = "lightrag-embedded"))]
            {
                let _ = config;
                Err(anyhow::anyhow!(
                    "lightrag-embedded requires the `lightrag-embedded` feature"
                ))
            }
        }
    }
}
