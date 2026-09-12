// SPDX-License-Identifier: MIT OR Apache-2.0
//! semdoc MCP server (stdio) — tools: query_semantic / query_text /
//! query_reranked / stats / list_documents.
//!
//! Schema-driven: the `filter` input is the Mongo-style JSON DSL, validated
//! against the database's schema at query time.

use anyhow::Result;
use clap::Parser;
use serde_json::{json, Value};
use std::io::Write;


#[derive(Parser)]
#[command(name = "semdoc-mcp", about = "MCP server for semdoc")]
struct Args {
    /// Database directory (contains schema.toml + LanceDB files)
    #[arg(short, long)]
    db: String,
    /// Enable reranked queries (loads the reranker backend from config/env)
    #[arg(long, default_value_t = false)]
    rerank: bool,
    /// Deployment config file (default: $SEMDOC_CONFIG or ./semdoc.config.toml)
    #[arg(long)]
    config: Option<String>,
}

// Business logic (Server/tools_list/dispatch) lives in semdoc::mcp so the
// HTTP transport in semdoc-server shares the exact same tool surface.
pub use semdoc::mcp::Server;

enum ReadResult {
    Eof,
    Empty,
    Msg(Value),
    ParseError(String),
}

fn read_message() -> ReadResult {
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => ReadResult::Eof,
        Ok(_) => {
            if line.trim().is_empty() {
                ReadResult::Empty
            } else {
                match serde_json::from_str(&line) {
                    Ok(v) => ReadResult::Msg(v),
                    Err(e) => ReadResult::ParseError(e.to_string()),
                }
            }
        }
        Err(e) => ReadResult::ParseError(e.to_string()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("semdoc MCP server starting (db: {})", args.db);
    let server = Server::new(&args.db, args.rerank, args.config.as_deref()).await?;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    loop {
        match read_message() {
            ReadResult::Eof => break,
            ReadResult::Empty => continue,
            ReadResult::ParseError(m) => {
                let resp = json!({
                    "jsonrpc": "2.0", "id": null,
                    "error": { "code": -32700, "message": format!("parse error: {m}") }
                });
                writeln!(out, "{}", resp)?;
                out.flush()?;
                continue;
            }
            ReadResult::Msg(msg) => {
                let id = msg.get("id").cloned().unwrap_or(Value::Null);
                let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
                let result = match method {
                    "initialize" => json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "semdoc", "version": "0.1.0" }
                    }),
                    "notifications/initialized" => json!(null),
                    "tools/list" => server.tools_list(),
                    "tools/call" => {
                        let params = msg.get("params").cloned().unwrap_or(json!({}));
                        let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
                        semdoc::mcp::dispatch(&server, name, &arguments).await
                    }
                    other => json!({
                        "error": { "code": -32601, "message": format!("method not found: {other}") }
                    }),
                };
                let resp = json!({ "jsonrpc": "2.0", "id": id, "result": result });
                writeln!(out, "{}", resp)?;
                out.flush()?;
            }
        }
    }
    Ok(())
}
