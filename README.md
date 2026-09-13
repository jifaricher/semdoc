<p align="center">
  <img src="docs/demo.gif" alt="semdoc 30-second demo: template init, ingest, semantic + full-text search" width="100%">
</p>

# semdocs — a schema-configurable RAG knowledge base, served over MCP

<p align="center">
  <a href="README_ZH.md">中文文档</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#mcp-use-with-claude-code">MCP / Claude Code</a> ·
  <a href="#hybrid-search">Hybrid search</a> ·
  <a href="#filtering">Filtering</a> ·
  <a href="https://crates.io/crates/semdocs"><img src="https://img.shields.io/crates/v/semdocs.svg" alt="crates.io"></a>
  <a href="https://github.com/jifaricher/semdoc/actions"><img src="https://img.shields.io/github/actions/workflow/status/jifaricher/semdoc/ci.yml?branch=dev" alt="CI"></a>
  <a href="LICENSE-APACHE"><img src="https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg" alt="license"></a>
</p>

**Define your entire RAG knowledge base in one TOML file — then use it
from Claude, Cursor or any MCP client as 11 ready-made tools.**

Most RAG stores hardcode your metadata columns (`doc_type`, `subsystem`, ...).
In semdoc the schema **is** a configuration file: declare the fields, vector
columns and graph plugins once — semdoc compiles the Mongo-style filter DSL,
the MCP tool schemas, the Web UI and physical-schema compatibility checks
from that single declaration. Same binary, different `schema.toml` → a
different product.

```bash
cargo install semdocs          # CLI + HTTP server + MCP server, one binary each
semdoc init --template code-search --db ./mydb
semdoc add --db ./mydb --file src/main.rs --meta language=rust
```

Then wire it into Claude Code (or any MCP client) in one line:

```bash
claude mcp add --transport http semdoc http://localhost:8092/mcp   --header "Authorization: Bearer $SEMDOC_MCP_TOKEN"
```

```toml
[vector]
fields = [
  { name = "dense_vec", dim = 1024, source = "raw_text", metric = "cosine", index = "none" },
]

[fields]                                     # YOUR schema, not ours
domain    = { type = "string", index = true }
topic     = { type = "string", index = true }
keywords  = { type = "list<string>" }
source_path = { type = "string", index = true, replace_key = true }

[plugins.graph]                              # optional multi-hop retrieval
backend = "lightrag-server"
endpoint = "http://127.0.0.1:9621"
```

## Quickstart (30 seconds)

```bash
cargo install --path .          # or: cargo build --release --bins

semdoc init --template code-search --db ./mydb   # 4 built-in templates
semdoc add   --db ./mydb --file src/main.rs --meta language=rust
semdoc query --db ./mydb -t "where do we parse the config" --limit 5
```

Four ways to talk to the same data:

| Interface | Use it for |
|---|---|
| `semdoc` CLI (`--db` local / `--server` remote) | ops, scripts, CI |
| `semdoc-mcp` (stdio) | local MCP clients |
| `semdoc-server /mcp` (streamable HTTP) | remote MCP clients, Claude Code |
| `semdoc-server /ui` | built-in read-only Web UI (browse / filter / search) |

## MCP: use with Claude Code

Add a running server to Claude Code in one line:

```bash
claude mcp add --transport http semdoc http://localhost:8092/mcp \
  --header "Authorization: Bearer $SEMDOC_MCP_TOKEN"
```

Or run the stdio server locally:

```json
{ "semdoc": { "command": "/path/to/semdoc-mcp", "args": ["--db", "/data/mydb"] } }
```

11 tools ship out of the box: `query_semantic`, `query_text`,
`query_reranked`, `query_graph`, `query_hybrid`, `get_document`,
`add_document`, `delete_document`, `update_document_metadata`,
`list_documents`, `stats` — each described in Chinese and English-aware
detail with schema-driven filter hints, so agents pick the right tool
without hand-holding. Sessions survive server restarts (SQLite-backed),
expire after 24h idle, and agents can auto-reconnect via a structured
`-32001` error envelope.

## Hybrid search

`query_hybrid` runs reranked vector search and LightRAG graph retrieval
in parallel and merges results by document id — graph entities give the
multi-hop structure, chunks give the grounding text, one call gets both.
Graph outages degrade to plain semantic search instead of failing.

## Filtering

Mongo-style JSON, validated against *your* schema and compiled to an
ANN pre-filter (filtering happens before the vector scan — no
over-fetch):

```json
{"subsystem": "mm", "priority": {"$gte": 3}, "keywords": {"$in": ["hugepage"]}}
```

Operators: `$eq $ne $in $nin $gt $gte $lt $lte $exists $and $or` —
values are type-checked against the schema, so `{"priority": "high"}`
is an error, not a silent no-match. Raw SQL exists as a local-only
escape hatch and is never exposed over the network.

## Docs

- [中文文档 README_ZH.md](README_ZH.md) — full reference: config files,
  MCP lifecycle, reserved columns, `replace_key` upserts, TLS/CA setup,
  lightrag integration
- `semdoc doctor` — one command checks config, embedder, reranker,
  graph, schema compatibility and indexes, each failure with a fix hint
- `docker compose up -d` — postgres (AGE+pgvector) and neo4j for the
  graph backend

## License

MIT OR Apache-2.0
