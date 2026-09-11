//! Schema configuration: `[table]`, `[vector]`, `[fields]`, `[plugins]`.
//!
//! One schema file describes one database. The parsed `SchemaConfig` is the
//! single source of truth from which the Arrow schema, the filter compiler,
//! the MCP tool schemas and the plugin set are all derived.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SchemaConfig {
    #[serde(default)]
    pub table: TableConfig,
    #[serde(default)]
    pub vector: VectorConfig,
    #[serde(default)]
    pub fields: BTreeMap<String, FieldConfig>,
    #[serde(default)]
    pub plugins: PluginsConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct TableConfig {
    /// LanceDB table name. Reserved: one table per database.
    #[serde(default = "default_table_name")]
    pub name: String,
}

fn default_table_name() -> String {
    "documents".to_string()
}

/// Vector column definition. A vector field is *derived*: `source` names the
/// text field whose content is embedded into this column at write time.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VectorFieldConfig {
    pub name: String,
    pub dim: usize,
    /// Which text field the embedding is derived from. Must reference a
    /// reserved text field (`raw_text`) or a user field of type `text`.
    pub source: String,
    #[serde(default = "default_metric")]
    pub metric: String, // cosine | l2 | dot
    /// `none` (brute force — right for small/medium KBs), `ivf_pq`, `ivf_flat`.
    #[serde(default = "default_vector_index")]
    pub index: String,
    /// Auto-embed on write using the default embedder. When false, only
    /// client-pre-embedded writes are accepted for this column.
    #[serde(default = "default_true")]
    pub auto_embed: bool,
}

fn default_metric() -> String {
    "cosine".to_string()
}
fn default_vector_index() -> String {
    "none".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct VectorConfig {
    /// Empty means "derive one vector field from raw_text using the default
    /// embedder's dimension" — resolved at init time into a single field
    /// named `dense_vec`.
    #[serde(default)]
    pub fields: Vec<VectorFieldConfig>,
}

/// User-defined scalar field.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FieldConfig {
    /// string | text | bool | int64 | float32 | timestamp | list<string>
    ///
    /// `string` is short, filterable metadata. `text` is long-form prose
    /// (searchable via FTS, usable as a vector `source`). `string` fields are
    /// NOT FTS-searchable; `text` fields are not directly filterable with
    /// equality (use FTS).
    pub r#type: String,
    /// Build a BTree scalar index for fast `only_if` pre-filtering.
    #[serde(default)]
    pub index: bool,
    /// Reject writes missing this field.
    #[serde(default)]
    pub required: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct PluginsConfig {
    #[serde(default)]
    pub graph: Option<GraphPluginConfig>,
    #[serde(default)]
    pub rerank: Option<RerankPluginConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
#[serde(tag = "backend")]
pub enum GraphPluginConfig {
    /// LightRAG server over HTTP — zero Python in the semdoc process.
    LightragServer {
        endpoint: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_query_timeout")]
        query_timeout_secs: u64,
        #[serde(default = "default_insert_concurrency")]
        insert_concurrency: usize,
        /// Background ainsert timeout (entity extraction is LLM-bound and
        /// slow; defaults generous like semrag's 1h).
        #[serde(default = "default_insert_timeout")]
        insert_timeout_secs: u64,
        #[serde(default = "default_delete_timeout")]
        delete_timeout_secs: u64,
    },
    /// In-process PyO3 lightrag (feature `lightrag-embedded`).
    LightragEmbedded {
        #[serde(default)]
        workspace: Option<String>,
        #[serde(default = "default_query_timeout")]
        query_timeout_secs: u64,
    },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
#[serde(tag = "backend")]
pub enum RerankPluginConfig {
    /// In-process INT8 ONNX cross-encoder (feature `onnx`).
    Onnx {
        #[serde(default)]
        dir: Option<String>,
    },
    /// TEI-compatible `/rerank` endpoint.
    Tei {
        endpoint: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default = "default_query_timeout")]
        timeout_secs: u64,
    },
    /// OpenAI-compatible `/v1/rerank` (base URL up to and including `/v1`).
    Openai {
        endpoint: String,
        #[serde(default)]
        api_key_env: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default = "default_query_timeout")]
        timeout_secs: u64,
    },
}

fn default_query_timeout() -> u64 {
    60
}
fn default_insert_timeout() -> u64 {
    600
}
fn default_delete_timeout() -> u64 {
    120
}
fn default_insert_concurrency() -> usize {
    4
}

/// Reserved (non-configurable) columns present in every database. Everything
/// else must come from `[fields]`.
pub const RESERVED_COLUMNS: &[&str] = &[
    "id",
    "raw_text",
    "chunk_level",
    "chunk_index",
    "parent_doc_id",
    "source_path",
    "source_type",
];

impl SchemaConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read schema {}: {e}", path.display()))?;
        let config: SchemaConfig = toml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse schema {}: {e}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        use std::collections::HashSet;

        // Vector fields: unique names, valid metric/index, resolvable source.
        let mut vec_names = HashSet::new();
        for v in &self.vector.fields {
            if !vec_names.insert(v.name.as_str()) {
                anyhow::bail!("duplicate vector field name: {}", v.name);
            }
            if RESERVED_COLUMNS.contains(&v.name.as_str()) {
                anyhow::bail!("vector field name `{}` collides with a reserved column", v.name);
            }
            if self.fields.contains_key(&v.name) {
                anyhow::bail!("vector field `{}` collides with a [fields] entry", v.name);
            }
            match v.dim {
                0 => anyhow::bail!("vector field `{}`: dim must be > 0", v.name),
                d if d > 100_000 => anyhow::bail!("vector field `{}`: dim {d} unrealistic", v.name),
                _ => {}
            }
            match v.metric.as_str() {
                "cosine" | "l2" | "dot" => {}
                other => anyhow::bail!(
                    "vector field `{}`: unknown metric `{other}` (cosine|l2|dot)",
                    v.name
                ),
            }
            match v.index.as_str() {
                "none" | "ivf_pq" | "ivf_flat" => {}
                other => anyhow::bail!(
                    "vector field `{}`: unknown index `{other}` (none|ivf_pq|ivf_flat)",
                    v.name
                ),
            }
            let source_ok = v.source == "raw_text"
                || self
                    .fields
                    .get(&v.source)
                    .is_some_and(|f| f.r#type == "text");
            if !source_ok {
                anyhow::bail!(
                    "vector field `{}`: source `{}` must be `raw_text` or a [fields] entry of type `text`",
                    v.name,
                    v.source
                );
            }
        }

        // Scalar fields: known types, no reserved collisions.
        for (name, f) in &self.fields {
            if RESERVED_COLUMNS.contains(&name.as_str()) {
                anyhow::bail!("field `{name}` collides with a reserved column");
            }
            match f.r#type.as_str() {
                "string" | "text" | "bool" | "int64" | "float32" | "timestamp"
                | "list<string>" => {}
                other => anyhow::bail!(
                    "field `{name}`: unknown type `{other}` \
                     (string|text|bool|int64|float32|timestamp|list<string>)"
                ),
            }
            if f.r#type == "text" && f.index {
                anyhow::bail!(
                    "field `{name}`: `text` fields are FTS-searched, not scalar-indexed \
                     (drop `index = true`)"
                );
            }
        }

        // Rerank backend vs build features.
        if let Some(r) = &self.plugins.rerank {
            match r {
                RerankPluginConfig::Onnx { .. } if !cfg!(feature = "onnx") => {
                    anyhow::bail!("rerank backend `onnx` requires building with feature `onnx`");
                }
                _ => {}
            }
        }
        if let Some(g) = &self.plugins.graph {
            if matches!(g, GraphPluginConfig::LightragEmbedded { .. })
                && !cfg!(feature = "lightrag-embedded")
            {
                anyhow::bail!(
                    "graph backend `lightrag-embedded` requires building with feature `lightrag-embedded`"
                );
            }
        }
        Ok(())
    }

    /// Default single-vector field used when `[vector].fields` is empty:
    /// one `dense_vec` column derived from `raw_text`. The dim is resolved
    /// from the loaded embedder at init time (written back into the manifest).
    pub fn effective_vector_fields(&self, default_dim: usize) -> Vec<VectorFieldConfig> {
        if !self.vector.fields.is_empty() {
            return self.vector.fields.clone();
        }
        vec![VectorFieldConfig {
            name: "dense_vec".to_string(),
            dim: default_dim,
            source: "raw_text".to_string(),
            metric: "cosine".to_string(),
            index: "none".to_string(),
            auto_embed: true,
        }]
    }

    /// User-defined text fields (eligible FTS columns / vector sources).
    pub fn text_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|(_, f)| f.r#type == "text")
            .map(|(n, _)| n.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal() {
        let c: SchemaConfig = toml::from_str("").unwrap();
        assert_eq!(c.table.name, "documents");
        assert!(c.vector.fields.is_empty());
        c.validate().unwrap();
    }

    #[test]
    fn parse_full() {
        let raw = r#"
[table]
name = "docs"

[[vector.fields]]
name = "dense_vec"
dim = 1024
source = "raw_text"

[[vector.fields]]
name = "title_vec"
dim = 768
source = "title"
metric = "l2"
index = "ivf_pq"

[fields]
category = { type = "string", index = true }
notes = { type = "text" }
score = { type = "float32", index = true }

[plugins.graph]
backend = "lightrag-server"
endpoint = "http://127.0.0.1:9727"

[plugins.rerank]
backend = "tei"
endpoint = "http://x:8000"
"#;
        let c: SchemaConfig = toml::from_str(raw).unwrap();
        c.validate().unwrap();
        assert_eq!(c.vector.fields.len(), 2);
        assert_eq!(c.fields.len(), 3);
        assert_eq!(c.text_fields(), vec!["notes"]);
    }

    #[test]
    fn reject_bad_source() {
        let c: SchemaConfig = toml::from_str(
            r#"
[[vector.fields]]
name = "v"
dim = 8
source = "nonexistent"
"#,
        )
        .unwrap();
        assert!(c.validate().is_err());
    }

    #[test]
    fn reject_reserved_collision() {
        let c: SchemaConfig = toml::from_str(
            r#"
[fields]
id = { type = "string" }
"#,
        )
        .unwrap();
        assert!(c.validate().is_err());
    }

    #[test]
    fn reject_text_with_scalar_index() {
        let c: SchemaConfig = toml::from_str(
            r#"
[fields]
notes = { type = "text", index = true }
"#,
        )
        .unwrap();
        assert!(c.validate().is_err());
    }
}
