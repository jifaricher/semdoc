# semdoc — config-schema-driven vector knowledge base

A general-purpose vector knowledge base where **the schema is a configuration
file**, not a hardcoded struct. One schema file → one database. Run the tool
with a different schema file → a different knowledge base. Storage, retrieval,
embedding, reranking and graph reasoning are all pluggable.

## Why

Traditional vector KBs hardcode their metadata columns (`doc_type`,
`subsystem`, ...). semdoc lets the user declare them:

```toml
[table]
name = "documents"

[vector]
fields = [
  { name = "dense_vec", dim = 1024, source = "raw_text", metric = "cosine", index = "none" },
]

[fields]
category = { type = "string", index = true }
score    = { type = "float32", index = true }
tags     = { type = "list<string>" }

[plugins.graph]
backend = "lightrag-server"
endpoint = "http://127.0.0.1:9727"

[plugins.rerank]
backend = "tei"
endpoint = "http://192.168.1.7:8000"
```

- **Vector fields** (`[vector]`) declare `name + dim + source + metric +
  index`. `source` names the text field the embedding is derived from —
  writes auto-embed; queries pick the column.
- **Scalar fields** (`[fields]`) are user-defined, optionally indexed
  (LanceDB BTree scalar index), and compile directly into ANN pre-filters —
  filtering happens *before* the vector search, so no over-fetch is needed.
- **Plugins** (`[plugins]`) wire in graph reasoning (lightrag) and rerank
  backends. A plugin that fails health checks degrades gracefully; the KB
  keeps working without it.

## Backends

| Capability | Backends |
|---|---|
| Embedding | `onnx` (bge-m3 in-process, feature `onnx`), `http` (OpenAI-compatible `/v1/embeddings`) |
| Reranker | `onnx` (INT8 cross-encoder), `tei` (HTTP `/rerank`), `openai` (`/v1/rerank`) — transient-failure retry + ANN-order degradation |
| Graph | `lightrag-server` (HTTP, zero Python), `lightrag-embedded` (PyO3, feature flag), `none` |

## Interfaces

- `semdoc` — CLI: `init` / `add` / `query` / `stats`
- `semdoc-server` — HTTP REST (`/documents`, `/query/*`) + MCP streamable
  HTTP (`/mcp`, bearer-token auth, persistent sessions)
- `semdoc-mcp` — MCP stdio server

MCP query results cap `raw_text` at 2000 chars (with a `truncated` marker)
and default to `expand_to = "chunk"` (small-to-big retrieval: ANN over
~512-char leaf chunks, expand on demand) so no tool result can blow the
client's token budget.

## Build

```bash
cargo build --release --bins
```

## Status

Work in progress.
