// SPDX-License-Identifier: MIT OR Apache-2.0
//! LanceDB storage with a **configuration-driven Arrow schema**.
//!
//! Fixed reserved columns (id / raw_text / chunk_level / chunk_index /
//! parent_doc_id / source_path / source_type) + `[fields]` scalar columns +
//! `[vector]` FixedSizeList columns, all derived from [`SchemaConfig`].
//!
//! Small-to-big retrieval: writes produce a parent row (chunk_level=0) plus
//! leaf chunk rows (chunk_level=1, blake3(parent || idx || text) ids);
//! queries recall leaves and expand per `ExpandTo`.

use anyhow::Result;
use arrow::array::{
    ArrayBuilder, BooleanBuilder, FixedSizeListBuilder, Float32Builder, Int64Builder,
    ListBuilder, StringBuilder, TimestampSecondBuilder, UInt32Builder, UInt8Builder,
};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit};
use lancedb::index::scalar::{BTreeIndexBuilder, FullTextSearchQuery, FtsIndexBuilder};
use lancedb::query::{ColumnOrdering, ExecutableQuery, QueryBase};
use lancedb::index::Index as LanceIndex;
use lancedb::Connection;
use serde_json::Value;
use std::sync::Arc;

use crate::schema::{FieldConfig, SchemaConfig, VectorFieldConfig};

pub const RESERVED_COLUMNS: &[&str] = &[
    "id",
    "raw_text",
    "chunk_level",
    "chunk_index",
    "parent_doc_id",
    "source_path",
    "source_type",
];

/// A document to store. `extra` carries the user-defined `[fields]` values
/// keyed by field name; vector columns are filled by the write pipeline
/// (auto-embed) or supplied directly by pre-embedded writes.
#[derive(Debug, Clone, Default)]
pub struct InputDoc {
    pub id: String,
    pub raw_text: String,
    pub source_path: String,
    pub source_type: String,
    pub language: Option<String>,
    pub extra: serde_json::Map<String, Value>,
    /// Pre-computed vectors keyed by vector field name. When missing for an
    /// `auto_embed` field, the write pipeline embeds `source` text.
    pub vectors: std::collections::HashMap<String, Vec<f32>>,
}

/// One stored row (leaf or parent). `extra` carries user fields.
#[derive(Debug, Clone)]
pub struct Record {
    pub id: String,
    pub raw_text: String,
    pub chunk_level: u8,
    pub chunk_index: u32,
    pub parent_doc_id: Option<String>,
    pub source_path: String,
    pub source_type: String,
    pub extra: serde_json::Map<String, Value>,
}

pub struct Store {
    pub conn: Connection,
    pub table: String,
    pub schema: Arc<Schema>,
    pub vector_fields: Vec<VectorFieldConfig>,
    pub scalar_fields: std::collections::BTreeMap<String, FieldConfig>,
}

impl Store {
    /// Public constructor for opening an *existing* database (no index
    /// creation) — used by CLI query paths and servers on read-only boot.
    pub fn new_existing(
        conn: Connection,
        table: String,
        schema: Arc<Schema>,
        vector_fields: Vec<VectorFieldConfig>,
        scalar_fields: std::collections::BTreeMap<String, FieldConfig>,
    ) -> Self {
        Self { conn, table, schema, vector_fields, scalar_fields }
    }

    /// Public batch builder.
    pub fn build_batch_public(
        &self,
        rows: &[(&Record, &std::collections::HashMap<String, Vec<f32>>)],
    ) -> Result<RecordBatch> {
        self.build_batch(rows)
    }

    /// Build the Arrow schema from the config. Reserved columns first
    /// (stable prefix), user scalar fields sorted by name (stable ordering
    /// across restarts), vector columns last (LanceDB convention).
    pub fn arrow_schema(config: &SchemaConfig, vector_fields: &[VectorFieldConfig]) -> Arc<Schema> {
        let mut fields: Vec<Field> = vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("raw_text", DataType::Utf8, false),
            Field::new("source_path", DataType::Utf8, false),
            Field::new("source_type", DataType::Utf8, false),
            Field::new("chunk_level", DataType::UInt8, false),
            Field::new("chunk_index", DataType::UInt32, false),
            Field::new("parent_doc_id", DataType::Utf8, true),
        ];
        for (name, fc) in &config.fields {
            fields.push(Field::new(name, scalar_arrow_type(&fc.r#type), true));
        }
        for v in vector_fields {
            fields.push(Field::new(
                &v.name,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    v.dim as i32,
                ),
                false,
            ));
        }
        Arc::new(Schema::new(Fields::from(fields)))
    }

    /// Open (and create if missing) the store for a config.
    pub async fn open(conn: Connection, config: &SchemaConfig, vector_fields: &[VectorFieldConfig]) -> Result<Self> {
        let table = config.table.name.clone();
        let schema = Self::arrow_schema(config, vector_fields);
        let tables = conn.table_names().execute().await?;
        if !tables.iter().any(|n| n == &table) {
            let empty = RecordBatch::new_empty(schema.clone());
            conn.create_table(&table, vec![empty]).execute().await?;
        }
        let store = Self {
            conn,
            table,
            schema,
            vector_fields: vector_fields.to_vec(),
            scalar_fields: config.fields.clone(),
        };
        store.create_indices(config).await?;
        Ok(store)
    }

    async fn create_indices(&self, config: &SchemaConfig) -> Result<()> {
        let t = self.conn.open_table(&self.table).execute().await?;
        // Scalar indexes: id + every `index = true` user field.
        t.create_index(&["id"], LanceIndex::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await?;
        for (name, fc) in &config.fields {
            if fc.index {
                t.create_index(&[name.as_str()], LanceIndex::BTree(BTreeIndexBuilder::default()))
                    .execute()
                    .await?;
            }
        }
        // FTS over raw_text + user `text` fields.
        let mut fts_cols = vec!["raw_text"];
        fts_cols.extend(config.text_fields());
        t.create_index(&fts_cols, LanceIndex::FTS(FtsIndexBuilder::default()))
            .execute()
            .await
            .or_else(|e| {
                // FTS index on the same columns twice errors; treat as ok.
                let msg = e.to_string();
                if msg.contains("already exists") { Ok(()) } else { Err(e) }
            })?;
        // Vector indexes per config (`index != "none"`).
        for v in &self.vector_fields {
            match v.index.as_str() {
                "ivf_pq" => {
                    t.create_index(
                        &[v.name.as_str()],
                        LanceIndex::IvfPq(lancedb::index::vector::IvfPqIndexBuilder::default()),
                    )
                    .execute()
                    .await?;
                }
                "ivf_flat" => {
                    t.create_index(
                        &[v.name.as_str()],
                        LanceIndex::IvfFlat(lancedb::index::vector::IvfFlatIndexBuilder::default()),
                    )
                    .execute()
                    .await?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Writes
    // -----------------------------------------------------------------------

    /// Upsert a parent + its leaf chunks via merge_insert on `id`.
    pub async fn upsert_doc(
        &self,
        parent: Record,
        leaves: Vec<Record>,
        parent_vectors: &std::collections::HashMap<String, Vec<f32>>,
        leaf_vectors: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        let mut rows: Vec<(&Record, &std::collections::HashMap<String, Vec<f32>>)> =
            vec![(&parent, parent_vectors)];
        for l in &leaves {
            rows.push((l, leaf_vectors));
        }
        let batch = self.build_batch(&rows)?;
        self.write_batch(batch).await
    }

    /// Public batch write: merge_insert on `id` (upsert semantics; the BTree
    /// index on `id` lets the merge planner do an index lookup).
    pub async fn write_batch(&self, batch: RecordBatch) -> Result<()> {
        use arrow_array::RecordBatchIterator;
        let table = self.conn.open_table(&self.table).execute().await?;
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), self.schema.clone()),
        );
        let mut builder = table.merge_insert(&["id"]);
        builder
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        builder.execute(reader).await?;
        Ok(())
    }

    fn build_batch(&self, rows: &[(&Record, &std::collections::HashMap<String, Vec<f32>>)]) -> Result<RecordBatch> {
        let n = rows.len();
        let mut builders = self.make_builders(n);

        for (rec, vectors) in rows {
            // Reserved columns
            set_str(builders.get_mut("id"), &rec.id);
            set_str(builders.get_mut("raw_text"), &rec.raw_text);
            set_str(builders.get_mut("source_path"), &rec.source_path);
            set_str(builders.get_mut("source_type"), &rec.source_type);
            set_str_opt(builders.get_mut("parent_doc_id"), rec.parent_doc_id.as_deref());
            set_u8(builders.get_mut("chunk_level"), rec.chunk_level);
            set_u32(builders.get_mut("chunk_index"), rec.chunk_index);

            // User scalar fields
            for (name, fc) in &self.scalar_fields {
                let v = rec.extra.get(name);
                set_scalar(builders.get_mut(name).ok_or_else(|| anyhow::anyhow!("no builder for {name}"))?, fc, v)?;
            }

            // Vector columns
            for vf in &self.vector_fields {
                let b = builders
                    .get_mut(&vf.name)
                    .ok_or_else(|| anyhow::anyhow!("no builder for vector {}", vf.name))?;
                let vec_b = b.as_any_mut().downcast_mut::<FixedSizeListBuilder<Float32Builder>>().unwrap();
                let values = vec_b.values();
                let vals = vectors
                    .get(&vf.name)
                    .ok_or_else(|| anyhow::anyhow!("missing vector for column {}", vf.name))?;
                if vals.len() != vf.dim as usize {
                    anyhow::bail!(
                        "vector column `{}`: expected dim {}, got {}",
                        vf.name,
                        vf.dim,
                        vals.len()
                    );
                }
                for &x in vals {
                    values.append_value(x);
                }
                vec_b.append(true);
            }
        }

        let mut cols: Vec<ArrayRef> = Vec::with_capacity(self.schema.fields().len());
        for f in self.schema.fields() {
            let name = f.name().as_str();
            let b = builders
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("missing builder for {name}"))?;
            cols.push(b.finish());
        }
        Ok(RecordBatch::try_new(self.schema.clone(), cols)?)
    }

    fn make_builders(&self, _capacity: usize) -> std::collections::BTreeMap<String, Box<dyn ArrayBuilder>> {
        let mut map: std::collections::BTreeMap<String, Box<dyn ArrayBuilder>> =
            std::collections::BTreeMap::new();
        for f in self.schema.fields() {
            let b: Box<dyn ArrayBuilder> = match f.data_type() {
                DataType::Utf8 => Box::new(StringBuilder::new()),
                DataType::UInt8 => Box::new(UInt8Builder::new()),
                DataType::UInt32 => Box::new(UInt32Builder::new()),
                DataType::Int64 => Box::new(Int64Builder::new()),
                DataType::Float32 => Box::new(Float32Builder::new()),
                DataType::Boolean => Box::new(BooleanBuilder::new()),
                DataType::Timestamp(TimeUnit::Second, _) => Box::new(TimestampSecondBuilder::new()),
                DataType::List(_) => Box::new(ListBuilder::new(StringBuilder::new())),
                DataType::FixedSizeList(_, size) => {
                    let values = Float32Builder::with_capacity((*size as usize) * 2);
                    Box::new(FixedSizeListBuilder::with_capacity(values, *size, 4))
                }
                other => panic!("unsupported column type {other:?}"),
            };
            map.insert(f.name().clone(), b);
        }
        map
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// ANN over leaf chunks (chunk_level=1, or pre-chunker NULL rows).
    pub async fn query_leaves(
        &self,
        vector_column: &str,
        embedding: &[f32],
        limit: usize,
        filter_sql: Option<&str>,
    ) -> Result<Vec<Record>> {
        self.ann_query(vector_column, embedding, limit, filter_sql, true).await
    }

    /// ANN over all rows (parents included).
    pub async fn query_all(
        &self,
        vector_column: &str,
        embedding: &[f32],
        limit: usize,
        filter_sql: Option<&str>,
    ) -> Result<Vec<Record>> {
        self.ann_query(vector_column, embedding, limit, filter_sql, false).await
    }

    async fn ann_query(
        &self,
        vector_column: &str,
        embedding: &[f32],
        limit: usize,
        filter_sql: Option<&str>,
        leaves_only: bool,
    ) -> Result<Vec<Record>> {
        let t = self.conn.open_table(&self.table).execute().await?;
        // Multi-vector-column schemas must name the ANN column explicitly
        // (lancedb errors with "More than one vector columns found" without
        // this); single-column schemas tolerate the explicit column too.
        let mut q = t
            .query()
            .nearest_to(embedding)?
            .column(vector_column);
        if leaves_only {
            let leaf = "(chunk_level = 1 OR chunk_level IS NULL)";
            let sql = match filter_sql {
                Some(f) => format!("{f} AND {leaf}"),
                None => leaf.to_string(),
            };
            q = q.only_if(sql);
        } else if let Some(f) = filter_sql {
            q = q.only_if(f.to_string());
        }
        q = q.limit(limit);
        // Bring back everything we need to reconstruct records.
        let mut projection: Vec<String> = vec![
            "id".into(),
            "raw_text".into(),
            "chunk_level".into(),
            "chunk_index".into(),
            "parent_doc_id".into(),
            "source_path".into(),
            "source_type".into(),
        ];
        for name in self.scalar_fields.keys() {
            projection.push(name.clone());
        }
        q = q.select(lancedb::query::Select::Columns(projection));
        let results = q.execute().await?;
        records_from_stream(results, &self.scalar_fields).await
    }

    /// FTS query over raw_text (user `text` fields are embedded into
    /// raw_text-bearing parents in this phase — FTS targets raw_text).
    pub async fn query_fts(&self, text: &str, limit: usize, filter_sql: Option<&str>) -> Result<Vec<Record>> {
        let t = self.conn.open_table(&self.table).execute().await?;
        let mut q = t
            .query()
            .full_text_search(FullTextSearchQuery::new(text.to_string()));
        if let Some(f) = filter_sql {
            q = q.only_if(f.to_string());
        }
        q = q.limit(limit);
        let results = q.execute().await?;
        records_from_stream(results, &self.scalar_fields).await
    }

    /// Fetch parent rows for the given ids.
    pub async fn get_parents(
        &self,
        ids: &[String],
    ) -> Result<std::collections::HashMap<String, Record>> {
        let mut out = std::collections::HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let unique: std::collections::BTreeSet<&String> = ids.iter().collect();
        let list = unique
            .iter()
            .map(|i| format!("'{}'", i.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        let t = self.conn.open_table(&self.table).execute().await?;
        let results = t
            .query()
            .only_if(format!("chunk_level = 0 AND id IN ({list})"))
            .limit(unique.len())
            .execute()
            .await?;
        for r in records_from_stream(results, &self.scalar_fields).await? {
            out.insert(r.id.clone(), r);
        }
        Ok(out)
    }

    pub async fn delete_by_id(&self, id: &str) -> Result<()> {
        let t = self.conn.open_table(&self.table).execute().await?;
        t.delete(&format!(
            "(id = '{0}' OR parent_doc_id = '{0}')",
            id.replace('\'', "''")
        ))
        .await?;
        Ok(())
    }

    /// Cap on the content-matching fallback scan. Beyond this, the mapping
    /// covers only the first N parents (relevance order lost anyway) and we
    /// warn — the real fix for large KBs is a content-hash side index.
    pub const ALL_PARENTS_CAP: usize = 50_000;

    /// Fetch parent rows (chunk_level=0), capped at ALL_PARENTS_CAP.
    pub async fn all_parents(&self) -> Result<Vec<Record>> {
        let t = self.conn.open_table(&self.table).execute().await?;
        let results = t
            .query()
            .only_if("chunk_level = 0")
            .limit(Self::ALL_PARENTS_CAP)
            .execute()
            .await?;
        let out = records_from_stream(results, &self.scalar_fields).await?;
        if out.len() >= Self::ALL_PARENTS_CAP {
            eprintln!(
                "[graph] all_parents hit cap {} — content-matching fallback may miss docs; \
                 consider a content-hash side index",
                Self::ALL_PARENTS_CAP
            );
        }
        Ok(out)
    }

    /// Fetch a single row by id (leaf or parent).
    pub async fn get_by_id(&self, id: &str) -> Result<Option<Record>> {
        let t = self.conn.open_table(&self.table).execute().await?;
        let results = t
            .query()
            .only_if(format!(
                "id = '{}'",
                id.replace('\'', "''")
            ))
            .limit(1)
            .execute()
            .await?;
        let recs = records_from_stream(results, &self.scalar_fields).await?;
        Ok(recs.into_iter().next())
    }

    /// Sibling chunks sharing a parent, sorted by chunk_index, excluding the
    /// hit itself — used by the `auto` sentence-window expansion.
    pub async fn get_sibling_chunks(
        &self,
        parent_doc_id: &str,
        exclude_chunk_index: u32,
        window: u32,
    ) -> Result<Vec<Record>> {
        let t = self.conn.open_table(&self.table).execute().await?;
        let pid = parent_doc_id.replace('\'', "''");
        let results = t
            .query()
            .only_if(format!(
                "chunk_level = 1 AND parent_doc_id = '{pid}' \
                 AND chunk_index >= {} AND chunk_index <= {} \
                 AND chunk_index != {exclude_chunk_index}",
                exclude_chunk_index.saturating_sub(window),
                exclude_chunk_index.saturating_add(window),
            ))
            .limit((window as usize * 2).max(2))
            .execute()
            .await?;
        let mut recs = records_from_stream(results, &self.scalar_fields).await?;
        recs.sort_by_key(|r| r.chunk_index);
        Ok(recs)
    }

    pub async fn count(&self) -> Result<usize> {
        let t = self.conn.open_table(&self.table).execute().await?;
        Ok(t.count_rows(None).await?)
    }

    /// Create the IVF indexes configured for vector columns (call after data
    /// exists — IVF needs enough rows to train centroids).
    /// Optimize/refresh all indices to cover newly written rows. Call after
    /// bulk writes (CLI `add`, server batch ingest). LanceDB index updates
    /// are incremental but not automatic for inverted (FTS) indexes created
    /// before the data existed.
    pub async fn optimize_indices(&self) -> Result<()> {
        let t = self.conn.open_table(&self.table).execute().await?;
        t.optimize(lancedb::table::OptimizeAction::default()).await?;
        Ok(())
    }

    pub async fn ensure_vector_indexes(&self) -> Result<()> {
        let t = self.conn.open_table(&self.table).execute().await?;
        for v in &self.vector_fields {
            match v.index.as_str() {
                "ivf_pq" => {
                    t.create_index(
                        &[v.name.as_str()],
                        LanceIndex::IvfPq(lancedb::index::vector::IvfPqIndexBuilder::default()),
                    )
                    .execute()
                    .await?;
                }
                "ivf_flat" => {
                    t.create_index(
                        &[v.name.as_str()],
                        LanceIndex::IvfFlat(lancedb::index::vector::IvfFlatIndexBuilder::default()),
                    )
                    .execute()
                    .await?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn scalar_arrow_type(t: &str) -> DataType {
    match t {
        "string" | "text" => DataType::Utf8,
        "bool" => DataType::Boolean,
        "int64" => DataType::Int64,
        "float32" => DataType::Float32,
        "timestamp" => DataType::Timestamp(TimeUnit::Second, None),
        "list<string>" => DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        other => panic!("validated schema has unknown type {other}"),
    }
}

fn downcast_sb(b: &mut Box<dyn ArrayBuilder>) -> &mut StringBuilder {
    b.as_any_mut().downcast_mut::<StringBuilder>().expect("utf8 builder")
}

fn set_str(b: Option<&mut Box<dyn ArrayBuilder>>, v: &str) {
    if let Some(b) = b {
        downcast_sb(b).append_value(v);
    }
}

fn set_str_opt(b: Option<&mut Box<dyn ArrayBuilder>>, v: Option<&str>) {
    if let Some(b) = b {
        match v {
            Some(s) => downcast_sb(b).append_value(s),
            None => downcast_sb(b).append_null(),
        }
    }
}

fn set_u8(b: Option<&mut Box<dyn ArrayBuilder>>, v: u8) {
    if let Some(b) = b {
        b.as_any_mut().downcast_mut::<UInt8Builder>().unwrap().append_value(v);
    }
}

fn set_u32(b: Option<&mut Box<dyn ArrayBuilder>>, v: u32) {
    if let Some(b) = b {
        b.as_any_mut().downcast_mut::<UInt32Builder>().unwrap().append_value(v);
    }
}

fn set_scalar(
    b: &mut Box<dyn ArrayBuilder>,
    fc: &FieldConfig,
    v: Option<&Value>,
) -> Result<()> {
    let missing_ok = !fc.required;
    let v = match v {
        Some(v) if !v.is_null() => v,
        _ => {
            if missing_ok {
                append_null(b, &fc.r#type);
                return Ok(());
            }
            anyhow::bail!("required field missing: {}", fc.r#type);
        }
    };
    match fc.r#type.as_str() {
        "string" | "text" => {
            let s = v.as_str().ok_or_else(|| anyhow::anyhow!("expected string"))?;
            downcast_sb(b).append_value(s);
        }
        "bool" => {
            let x = v.as_bool().ok_or_else(|| anyhow::anyhow!("expected bool"))?;
            b.as_any_mut().downcast_mut::<BooleanBuilder>().unwrap().append_value(x);
        }
        "int64" => {
            let x = v.as_i64().ok_or_else(|| anyhow::anyhow!("expected int64"))?;
            b.as_any_mut().downcast_mut::<Int64Builder>().unwrap().append_value(x);
        }
        "float32" => {
            let x = v.as_f64().ok_or_else(|| anyhow::anyhow!("expected float32"))?;
            b.as_any_mut().downcast_mut::<Float32Builder>().unwrap().append_value(x as f32);
        }
        "timestamp" => {
            let x = v.as_i64().ok_or_else(|| anyhow::anyhow!("expected timestamp (unix secs)"))?;
            b.as_any_mut().downcast_mut::<TimestampSecondBuilder>().unwrap().append_value(x);
        }
        "list<string>" => {
            let arr = v.as_array().ok_or_else(|| anyhow::anyhow!("expected array"))?;
            let lb = b.as_any_mut().downcast_mut::<ListBuilder<StringBuilder>>().unwrap();
            for item in arr {
                lb.values().append_value(
                    item.as_str().ok_or_else(|| anyhow::anyhow!("expected list<string>"))?,
                );
            }
            lb.append(true);
        }
        other => anyhow::bail!("unsupported field type {other}"),
    }
    Ok(())
}

fn append_null(b: &mut Box<dyn ArrayBuilder>, t: &str) {
    match t {
        "string" | "text" => downcast_sb(b).append_null(),
        "bool" => b.as_any_mut().downcast_mut::<BooleanBuilder>().unwrap().append_null(),
        "int64" => b.as_any_mut().downcast_mut::<Int64Builder>().unwrap().append_null(),
        "float32" => b.as_any_mut().downcast_mut::<Float32Builder>().unwrap().append_null(),
        "timestamp" => b.as_any_mut().downcast_mut::<TimestampSecondBuilder>().unwrap().append_null(),
        "list<string>" => b
            .as_any_mut()
            .downcast_mut::<ListBuilder<StringBuilder>>()
            .unwrap()
            .append(false),
        _ => {}
    }
}

async fn records_from_stream(
    mut stream: lancedb::arrow::SendableRecordBatchStream,
    scalar_fields: &std::collections::BTreeMap<String, FieldConfig>,
) -> Result<Vec<Record>> {
    use futures::TryStreamExt;
    let mut out = Vec::new();
    while let Some(batch) = stream.try_next().await
        .map_err(|e| anyhow::anyhow!("lance stream: {e}"))?
    {
        let ids = str_col(&batch, "id")?;
        let raw = str_col(&batch, "raw_text")?;
        let levels = batch
            .column_by_name("chunk_level")
            .ok_or_else(|| anyhow::anyhow!("missing chunk_level"))?;
        let levels = levels
            .as_any()
            .downcast_ref::<arrow_array::UInt8Array>()
            .ok_or_else(|| anyhow::anyhow!("chunk_level not u8"))?;
        let indexes = batch
            .column_by_name("chunk_index")
            .ok_or_else(|| anyhow::anyhow!("missing chunk_index"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| anyhow::anyhow!("chunk_index not u32"))?
            .clone();
        let parents = str_col(&batch, "parent_doc_id")?;
        let paths = str_col(&batch, "source_path")?;
        let stypes = str_col(&batch, "source_type")?;

        for i in 0..batch.num_rows() {
            let mut extra = serde_json::Map::new();
            for (name, _) in scalar_fields {
                if let Some(col) = batch.column_by_name(name) {
                    if col.is_null(i) {
                        continue;
                    }
                    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                        extra.insert(name.clone(), Value::String(a.value(i).to_string()));
                    } else if let Some(a) = col.as_any().downcast_ref::<arrow_array::BooleanArray>() {
                        extra.insert(name.clone(), Value::Bool(a.value(i)));
                    } else if let Some(a) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                        extra.insert(name.clone(), Value::Number(a.value(i).into()));
                    } else if let Some(a) = col.as_any().downcast_ref::<arrow_array::Float32Array>() {
                        extra.insert(
                            name.clone(),
                            serde_json::Number::from_f64(a.value(i) as f64)
                                .map(Value::Number)
                                .unwrap_or(Value::Null),
                        );
                    } else if let Some(a) = col.as_any().downcast_ref::<arrow_array::TimestampSecondArray>()
                    {
                        extra.insert(name.clone(), Value::Number(a.value(i).into()));
                    } else if let Some(a) =
                        col.as_any().downcast_ref::<arrow_array::ListArray>()
                    {
                        let mut items = Vec::new();
                        let vals = a.value(i);
                        for j in 0..vals.len() {
                            if vals.is_null(j) {
                                continue;
                            }
                            if let Some(s) = vals.as_any().downcast_ref::<StringArray>() {
                                items.push(Value::String(s.value(j).to_string()));
                            }
                        }
                        extra.insert(name.clone(), Value::Array(items));
                    }
                }
            }
            out.push(Record {
                id: ids.value(i).to_string(),
                raw_text: raw.value(i).to_string(),
                chunk_level: levels.value(i),
                chunk_index: indexes.value(i),
                parent_doc_id: (!parents.is_null(i)).then(|| parents.value(i).to_string()),
                source_path: paths.value(i).to_string(),
                source_type: stypes.value(i).to_string(),
                extra,
            });
        }
    }
    Ok(out)
}

fn str_col(batch: &RecordBatch, name: &str) -> Result<StringArray> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("missing column {name}"))?;
    if col.is_nullable() {
        // Cast nullable to owned array; nulls allowed only for parent_doc_id.
        let arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("column {name} not utf8"))?;
        Ok(arr.clone())
    } else {
        let arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("column {name} not utf8"))?;
        Ok(arr.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{FieldConfig, SchemaConfig};

    fn two_field_schema() -> SchemaConfig {
        SchemaConfig {
            table: Default::default(),
            vector: crate::schema::VectorConfig {
                fields: vec![VectorFieldConfig {
                    name: "dense_vec".into(),
                    dim: 4,
                    source: "raw_text".into(),
                    metric: "cosine".into(),
                    index: "none".into(),
                    auto_embed: true,
                }],
            },
            fields: [
                ("category".to_string(), FieldConfig { r#type: "string".into(), index: true, required: false }),
                ("keywords".to_string(), FieldConfig { r#type: "list<string>".into(), index: false, required: false }),
            ]
            .into_iter()
            .collect(),
            plugins: Default::default(),
        }
    }

    async fn test_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let conn = lancedb::connect(dir.path().to_str().unwrap())
            .execute()
            .await
            .unwrap();
        let cfg = two_field_schema();
        let vfs = cfg.effective_vector_fields(4);
        let store = Store::open(conn, &cfg, &vfs).await.unwrap();
        (dir, store)
    }

    fn parent_record(id: &str, text: &str, extra: serde_json::Map<String, Value>) -> Record {
        Record {
            id: id.into(),
            raw_text: text.into(),
            chunk_level: 0,
            chunk_index: 0,
            parent_doc_id: None,
            source_path: format!("/docs/{id}.md"),
            source_type: "file".into(),
            extra,
        }
    }

    fn leaf_record(parent_id: &str, idx: u32, text: &str) -> Record {
        Record {
            id: crate::chunker::Chunk::compute_id(parent_id, idx, text),
            raw_text: text.into(),
            chunk_level: 1,
            chunk_index: idx,
            parent_doc_id: Some(parent_id.into()),
            source_path: String::new(),
            source_type: "file".into(),
            extra: Default::default(),
        }
    }

    fn unit_vec(seed: f32) -> Vec<f32> {
        vec![seed, 1.0 - seed, seed * 0.5, 0.25]
    }

    fn leaf_vectors(seed: f32) -> std::collections::HashMap<String, Vec<f32>> {
        [("dense_vec".to_string(), unit_vec(seed))].into_iter().collect()
    }

    #[tokio::test]
    async fn upsert_and_count_roundtrip() {
        let (_dir, store) = test_store().await;
        let mut extra = serde_json::Map::new();
        extra.insert("category".into(), Value::String("mm".into()));
        extra.insert("keywords".into(), serde_json::json!(["a", "b"]));

        store
            .upsert_doc(
                parent_record("p1", "parent text", extra),
                vec![leaf_record("p1", 0, "leaf zero"), leaf_record("p1", 1, "leaf one")],
                &leaf_vectors(1.0),
                &leaf_vectors(0.0),
            )
            .await
            .unwrap();
        assert_eq!(store.count().await.unwrap(), 3);

        // merge_insert touches only the rows present in the batch: parent +
        // leaf 0 are updated in place, but an omitted leaf (leaf 1) stays
        // behind — stale-chunk cleanup is the caller's job (delete_by_id).
        store
            .upsert_doc(
                parent_record("p1", "parent text v2", Default::default()),
                vec![leaf_record("p1", 0, "leaf zero")],
                &leaf_vectors(1.0),
                &leaf_vectors(0.0),
            )
            .await
            .unwrap();
        assert_eq!(store.count().await.unwrap(), 3);

        let p = store.get_by_id("p1").await.unwrap().unwrap();
        assert_eq!(p.raw_text, "parent text v2");
        assert_eq!(p.chunk_level, 0);
        // The stale leaf is still queryable.
        let stale = store
            .get_by_id(&leaf_record("p1", 1, "leaf one").id)
            .await
            .unwrap();
        assert!(stale.is_some());
    }

    #[tokio::test]
    async fn roundtrip_scalar_and_list_fields() {
        let (_dir, store) = test_store().await;
        let mut extra = serde_json::Map::new();
        extra.insert("category".into(), Value::String("sched".into()));
        extra.insert("keywords".into(), serde_json::json!(["hugepage", "mlock"]));

        store
            .upsert_doc(
                parent_record("p2", "text", extra),
                vec![],
                &leaf_vectors(0.5),
                &Default::default(),
            )
            .await
            .unwrap();
        let rec = store.get_by_id("p2").await.unwrap().unwrap();
        assert_eq!(rec.extra["category"], Value::String("sched".into()));
        assert_eq!(
            rec.extra["keywords"],
            serde_json::json!(["hugepage", "mlock"])
        );
    }

    #[tokio::test]
    async fn wrong_dim_vector_is_rejected() {
        let (_dir, store) = test_store().await;
        let bad: std::collections::HashMap<String, Vec<f32>> =
            [("dense_vec".to_string(), vec![1.0, 2.0])].into_iter().collect();
        let err = store
            .upsert_doc(parent_record("p3", "t", Default::default()), vec![], &bad, &Default::default())
            .await;
        assert!(err.is_err());
        assert!(format!("{:#}", err.unwrap_err()).contains("expected dim 4"));
    }

    #[tokio::test]
    async fn ann_query_leaves_vs_all() {
        let (_dir, store) = test_store().await;
        let mut extra = serde_json::Map::new();
        extra.insert("category".into(), Value::String("mm".into()));
        let mut leaf_extra = serde_json::Map::new();
        leaf_extra.insert("category".into(), Value::String("mm".into()));
        let mut l0 = leaf_record("p1", 0, "leaf a");
        l0.extra = leaf_extra.clone();
        let mut l1 = leaf_record("p1", 1, "leaf b");
        l1.extra = leaf_extra;
        store
            .upsert_doc(parent_record("p1", "parent", extra), vec![l0, l1], &leaf_vectors(1.0), &leaf_vectors(1.0))
            .await
            .unwrap();

        // Leaves only: parents excluded.
        let leaves = store
            .query_leaves("dense_vec", &unit_vec(1.0), 10, None)
            .await
            .unwrap();
        assert_eq!(leaves.len(), 2);
        assert!(leaves.iter().all(|r| r.chunk_level == 1));
        assert_eq!(leaves[0].extra["category"], Value::String("mm".into()));

        // All rows: parents included.
        let all = store
            .query_all("dense_vec", &unit_vec(1.0), 10, None)
            .await
            .unwrap();
        assert_eq!(all.len(), 3);

        // Scalar pre-filter via SQL.
        let filtered = store
            .query_leaves("dense_vec", &unit_vec(1.0), 10, Some("category = 'mm'"))
            .await
            .unwrap();
        assert_eq!(filtered.len(), 2);
        let none = store
            .query_leaves("dense_vec", &unit_vec(1.0), 10, Some("category = 'net'"))
            .await
            .unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn get_parents_dedups_and_maps() {
        let (_dir, store) = test_store().await;
        store
            .upsert_doc(
                parent_record("p1", "parent", Default::default()),
                vec![leaf_record("p1", 0, "a"), leaf_record("p1", 1, "b")],
                &leaf_vectors(1.0),
                &leaf_vectors(0.0),
            )
            .await
            .unwrap();
        let leaves = store.get_by_id(leaf_record("p1", 0, "a").id.as_str()).await.unwrap();
        assert!(leaves.is_some());

        // Duplicate ids are deduped; missing ids are absent from the map.
        let parents = store
            .get_parents(&["p1".into(), "p1".into(), "missing".into()])
            .await
            .unwrap();
        assert_eq!(parents.len(), 1);
        assert_eq!(parents["p1"].raw_text, "parent");
    }

    #[tokio::test]
    async fn sibling_chunks_window_excludes_self() {
        let (_dir, store) = test_store().await;
        store
            .upsert_doc(
                parent_record("p1", "parent", Default::default()),
                vec![
                    leaf_record("p1", 0, "c0"),
                    leaf_record("p1", 1, "c1"),
                    leaf_record("p1", 2, "c2"),
                    leaf_record("p1", 3, "c3"),
                ],
                &leaf_vectors(1.0),
                &leaf_vectors(0.0),
            )
            .await
            .unwrap();
        let sibs = store.get_sibling_chunks("p1", 1, 1).await.unwrap();
        let idxs: Vec<u32> = sibs.iter().map(|r| r.chunk_index).collect();
        assert_eq!(idxs, vec![0, 2]);
        // Far hit: window excludes everything.
        let sibs = store.get_sibling_chunks("p1", 0, 0).await.unwrap();
        assert!(sibs.is_empty());
    }

    #[tokio::test]
    async fn delete_by_id_cascades_to_leaves() {
        let (_dir, store) = test_store().await;
        store
            .upsert_doc(
                parent_record("p1", "parent", Default::default()),
                vec![leaf_record("p1", 0, "a"), leaf_record("p1", 1, "b")],
                &leaf_vectors(1.0),
                &leaf_vectors(0.0),
            )
            .await
            .unwrap();
        store
            .upsert_doc(
                parent_record("p2", "other", Default::default()),
                vec![leaf_record("p2", 0, "c")],
                &leaf_vectors(0.2),
                &leaf_vectors(0.2),
            )
            .await
            .unwrap();
        assert_eq!(store.count().await.unwrap(), 5);

        store.delete_by_id("p1").await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);
        assert!(store.get_by_id("p1").await.unwrap().is_none());
        assert!(store.get_by_id("p2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn fts_finds_substring_terms() {
        let (_dir, store) = test_store().await;
        store
            .upsert_doc(
                parent_record("p1", "the scheduler uses EEVDF heuristics", Default::default()),
                vec![],
                &leaf_vectors(1.0),
                &Default::default(),
            )
            .await
            .unwrap();
        store
            .upsert_doc(
                parent_record("p2", "memory folio batching", Default::default()),
                vec![],
                &leaf_vectors(0.5),
                &Default::default(),
            )
            .await
            .unwrap();

        let hits = store.query_fts("eevdf", 10, None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "p1");

        let hits = store.query_fts("folio", 10, None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "p2");
    }

    #[tokio::test]
    async fn all_parents_returns_only_parents() {
        let (_dir, store) = test_store().await;
        store
            .upsert_doc(
                parent_record("p1", "parent", Default::default()),
                vec![leaf_record("p1", 0, "a")],
                &leaf_vectors(1.0),
                &leaf_vectors(0.0),
            )
            .await
            .unwrap();
        let parents = store.all_parents().await.unwrap();
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0].chunk_level, 0);
    }
}
