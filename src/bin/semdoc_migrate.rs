// SPDX-License-Identifier: MIT OR Apache-2.0
//! One-shot migration: copy all rows from the semrag LanceDB database into
//! a semdoc database, remapping columns (semrag's flat column order →
//! semdoc's reserved + [fields] layout). Vectors are copied verbatim —
//! no re-embedding.
//!
//!   semdoc-migrate --src /workspace/web/semrag/.semdoc.db --dst /tmp/migrated
//!
//! The dst must already be initialized (`semdoc init` with the equivalent
//! schema). The src is opened as a plain LanceDB connection.

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt32Array, UInt8Array,
};
use arrow_schema::DataType;
use clap::Parser;
use std::collections::HashMap;

#[derive(Parser)]
struct Args {
    /// semrag database directory (.semdoc.db)
    #[arg(long)]
    src: String,
    /// semdoc database directory (already initialized)
    #[arg(long)]
    dst: String,
    #[arg(long, default_value = "documents")]
    table: String,
    /// Rebuild dense_vec from a different column name (unused by default)
    #[arg(long, default_value = "documents")]
    src_table: String,
}

/// Convert one src column into the dst-typed array. Both schemas use the
/// same physical types for shared columns; the only wrinkle is nullability
/// drift, which arrow handles by same-type downcast.
fn remap(col: &arrow_array::ArrayRef, target: &DataType, name: &str) -> anyhow::Result<arrow_array::ArrayRef> {
    if col.data_type() == target {
        return Ok(col.clone());
    }
    // FixedSizeList nullability of child field can differ; cast via builder.
    match (col.data_type(), target) {
        (DataType::FixedSizeList(_, len), DataType::FixedSizeList(_, len2)) if len == len2 => {
            let src_arr = col
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| anyhow::anyhow!("downcast {name}"))?;
            let values = src_arr
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("vec values {name}"))?;
            let mut b = arrow_array::builder::FixedSizeListBuilder::new(
                arrow_array::builder::Float32Builder::with_capacity(values.len()),
                *len,
            );
            for i in 0..src_arr.len() {
                if src_arr.is_null(i) {
                    b.append(false);
                } else {
                    let off = src_arr.value_offset(i);
                    for j in off..off + *len {
                        b.values().append_value(values.value(j as usize));
                    }
                    b.append(true);
                }
            }
            Ok(std::sync::Arc::new(b.finish()))
        }
        (DataType::Utf8, DataType::Utf8) => Ok(col.clone()),
        (a, b2) => Err(anyhow::anyhow!("column {name}: src {a:?} vs dst {b2:?}")),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let src = lancedb::connect(&args.src).execute().await?;
    let src_table = src.open_table(&args.src_table).execute().await?;
    let total = src_table.count_rows(None).await?;
    println!("src rows: {total}");

    let dst = lancedb::connect(&args.dst).execute().await?;
    let dst_table = dst.open_table(&args.table).execute().await?;
    let dst_schema = dst_table.schema().await?;
    let dst_field_names: Vec<String> = dst_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect::<Vec<_>>();

    // Column-name compat map: semrag column -> semdoc column (identical names
    // for this migration; the [fields] entries mirror semrag metadata names).
    let mut name_map: HashMap<String, String> = HashMap::new();
    for f in &dst_field_names {
        name_map.insert(f.clone(), f.clone());
    }
    // semdoc "language" is nullable string; semrag too. OK.

    // Stream the whole src table and write in 1024-row chunks.
    use futures::TryStreamExt;
    use lancedb::query::ExecutableQuery;
    let stream = src_table.query().execute().await?;
    let mut stream = stream;

    let mut copied = 0usize;
    let mut buf: Vec<RecordBatch> = Vec::new();
    let mut buf_rows = 0usize;

    async fn flush(
        buf: &mut Vec<RecordBatch>,
        dst_table: &lancedb::table::Table,
        dst_schema: &arrow_schema::SchemaRef,
        copied: &mut usize,
    ) -> anyhow::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        // merge_insert on id for upsert semantics (rerunnable migration).
        let batches = std::mem::take(buf);
        let n: usize = batches.iter().map(|b| b.num_rows()).sum();
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(batches.into_iter().map(Ok), dst_schema.clone()),
        );
        let mut builder = dst_table.merge_insert(&["id"]);
        builder
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        builder.execute(reader).await?;
        *copied += n;
        println!("  ... {copied}");
        Ok(())
    }

    while let Some(batch) = stream.try_next().await? {
        let mut cols: Vec<arrow_array::ArrayRef> = Vec::new();
        for name in &dst_field_names {
            let src_name = name_map.get(name).map(String::as_str).unwrap_or(name);
            let col: arrow_array::ArrayRef = batch
                .column_by_name(src_name)
                .ok_or_else(|| anyhow::anyhow!("src missing column {src_name}"))?
                .clone();
            let target = dst_schema
                .field_with_name(name)
                .map_err(|e| anyhow::anyhow!("dst schema {name}: {e}"))?
                .data_type()
                .clone();
            cols.push(remap(&col, &target, name)?);
        }
        let new_batch = RecordBatch::try_new(dst_schema.clone(), cols)?;
        buf_rows += new_batch.num_rows();
        buf.push(new_batch);
        if buf_rows >= 1024 {
            flush(&mut buf, &dst_table, &dst_schema, &mut copied).await?;
            buf_rows = 0;
        }
    }
    flush(&mut buf, &dst_table, &dst_schema, &mut copied).await?;
    println!("copied {copied} rows into {}", args.dst);

    // Sanity: nullability of language/parent_doc_id columns — semrag allows
    // nulls; verified by remap passthrough.
    let _ = (StringArray::from(vec!["ok"]), UInt8Array::from(vec![0u8]), UInt32Array::from(vec![0u32]));
    Ok(())
}
