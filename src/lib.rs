// SPDX-License-Identifier: MIT OR Apache-2.0
//! semdoc — config-schema-driven vector knowledge base.
//!
//! One schema file (`schema.toml`) describes one database:
//! - `[table]`   — storage name
//! - `[vector]`  — vector columns (name/dim/source/metric/index)
//! - `[fields]`  — user-defined scalar columns (typed, optionally indexed)
//! - `[plugins]` — graph reasoning + rerank backends
//!
//! All retrieval logic derives from the parsed [`schema::SchemaConfig`].

pub mod chunker;
pub mod config;
pub mod embedding;
pub mod mcp;
pub mod plugins;
pub mod query;
pub mod reranker;
pub mod schema;
pub mod store;

pub use schema::SchemaConfig;
