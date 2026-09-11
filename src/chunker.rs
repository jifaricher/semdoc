// SPDX-License-Identifier: MIT OR Apache-2.0
//! Markdown-aware + plain-recursive chunker for the atomic LanceDB store.
//!
//! Two-stage splitting, modelled on LangChain's `MarkdownHeaderTextSplitter` +
//! `RecursiveCharacterTextSplitter` combo and LlamaIndex's hierarchical node
//! parsers. Output is a flat `Vec<Chunk>` suitable for one LanceDB write per
//! chunk plus a single parent-doc row written by the caller.
//!
//! Why this exists separately from lightrag's own chunker: lightrag chunks
//! for *LLM entity/relation extraction* (~1200 chars, token-budgeted);
//! LanceDB chunks for *recall precision + metadata-filtered retrieval*
//! (~512 chars). The two stores have different downstream consumers so
//! their chunk sizes are tuned independently.

use blake3::hash;

/// Default target chunk size in Unicode chars. Matches LlamaIndex's
/// mid-level HierarchicalNodeParser default (512) and LangChain examples
/// scaled up ~2×. Tuned for Chinese markdown where 1 paragraph is
/// typically 200-600 chars.
pub const DEFAULT_CHUNK_SIZE: usize = 512;

/// Overlap between consecutive chunks in chars. ~10% of chunk_size —
/// standard RAG practice to preserve cross-boundary context.
pub const DEFAULT_CHUNK_OVERLAP: usize = 50;

/// Env var name for chunk size override.
pub const CHUNK_SIZE_ENV: &str = "SEMDOC_CHUNK_SIZE";

/// Env var name for chunk overlap override.
pub const CHUNK_OVERLAP_ENV: &str = "SEMDOC_CHUNK_OVERLAP";

/// Resolve chunk size from `SEMDOC_CHUNK_SIZE` env var, falling back to
/// `DEFAULT_CHUNK_SIZE`. Reads once per call so callers changing env at
/// runtime (e.g. tests) see the new value without restart. Clamped to
/// `[64, 8192]` — below 64 chars chunks become noise (single sentence
/// fragments), above 8192 they exceed bge-m3's token budget and the
/// embedding starts diluting signal.
pub fn resolve_chunk_size() -> usize {
    std::env::var(CHUNK_SIZE_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 64 && n <= 8192)
        .unwrap_or(DEFAULT_CHUNK_SIZE)
}

/// Resolve overlap from `SEMDOC_CHUNK_OVERLAP` env var, falling back to
/// `DEFAULT_CHUNK_OVERLAP`. Clamped to `[0, chunk_size/2]` — overlap
/// larger than half the chunk size produces chunks that overlap more
/// than they advance, which is wasteful.
pub fn resolve_chunk_overlap(chunk_size: usize) -> usize {
    let max = chunk_size / 2;
    std::env::var(CHUNK_OVERLAP_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n <= max)
        .unwrap_or(DEFAULT_CHUNK_OVERLAP.min(max))
}

/// One output chunk. `chunk_index` is the 0-based position in the parent
/// document; `chunk_id` is a deterministic blake3 of (parent_id, index,
/// text) so the same parent re-ingested with the same text produces the
/// same chunk ids → LanceDB `merge_insert` upserts in place instead of
/// duplicating rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    pub parent_doc_id: String,
    pub chunk_index: u32,
    pub chunk_id: String,
}

impl Chunk {
    /// blake3(parent_id || ":" || chunk_index || ":" || text) → hex string.
    /// Concatenated with `:` separators so different parents can't collide
    /// even if their text overlaps, and chunk_index is fixed-width-padded
    /// to avoid `index 1` matching `index 10` prefix.
    pub fn compute_id(parent_doc_id: &str, chunk_index: u32, text: &str) -> String {
        let mut input = String::with_capacity(parent_doc_id.len() + 16 + text.len() + 2);
        input.push_str(parent_doc_id);
        input.push(':');
        input.push_str(&chunk_index.to_string());
        input.push(':');
        input.push_str(text);
        hash(input.as_bytes()).to_hex().to_string()
    }
}

/// Split markdown text into chunks. Two-stage: first split on `##` / `###`
/// / `####` headers (preserving the header line at the start of each
/// section), then run each section through `recursive_char_split` so an
/// oversized section gets broken down to `chunk_size` but a tight section
/// stays together.
///
/// `#` (h1) is intentionally NOT a split point — many kernel docs use a
/// single `# Title` at the top and we don't want a separate chunk that's
/// just the title. `##` and deeper are the real section boundaries.
pub fn chunk_markdown(text: &str, parent_doc_id: &str) -> Vec<Chunk> {
    chunk_markdown_with(text, parent_doc_id, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
}

pub fn chunk_markdown_with(
    text: &str,
    parent_doc_id: &str,
    chunk_size: usize,
    overlap: usize,
) -> Vec<Chunk> {
    let sections = split_markdown_by_headers(text);
    // Coalesce: short sections (≤ chunk_size/3) get merged into the
    // NEXT section rather than stand alone. This avoids the "1-line
    // header is its own chunk" failure mode where the eevdf doc's
    // title+preamble (`# EEVDF...`) would split off as chunk[0] of 18
    // chars, or `### 入队/出队流程` alone as 11 chars.
    //
    // Walk sections in reverse so we can pull a short section forward
    // into the next one without re-checking the merged result.
    let coalesce_threshold = (chunk_size / 3).max(64);
    let mut coalesced: Vec<String> = Vec::new();
    for section in sections.into_iter().rev() {
        let len = section.chars().count();
        if len < coalesce_threshold {
            if let Some(next) = coalesced.last_mut() {
                // Prepend the short section to the next (which is the
                // following section in original order, since we're
                // walking reversed).
                let mut merged = section.clone();
                merged.push_str("\n\n");
                merged.push_str(next);
                *next = merged;
                continue;
            }
            // No next section to merge into — keep it (it's the trailing
            // short tail of the doc).
        }
        coalesced.push(section);
    }
    coalesced.reverse();

    let mut chunks: Vec<(String, String)> = Vec::new();
    for section in coalesced {
        if section.chars().count() <= chunk_size {
            chunks.push((section, String::new()));
        } else {
            for piece in recursive_char_split(&section, chunk_size, overlap) {
                if !piece.trim().is_empty() {
                    chunks.push((piece, String::new()));
                }
            }
        }
    }

    finalize_chunks(chunks, parent_doc_id)
}

/// Split plain (non-markdown) text via recursive char split only.
pub fn chunk_plain(text: &str, parent_doc_id: &str) -> Vec<Chunk> {
    chunk_plain_with(text, parent_doc_id, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
}

pub fn chunk_plain_with(
    text: &str,
    parent_doc_id: &str,
    chunk_size: usize,
    overlap: usize,
) -> Vec<Chunk> {
    let pieces = recursive_char_split(text, chunk_size, overlap);
    let chunks: Vec<(String, String)> = pieces
        .into_iter()
        .filter(|p| !p.trim().is_empty())
        .map(|p| (p, String::new()))
        .collect();
    finalize_chunks(chunks, parent_doc_id)
}

/// Decide whether to use markdown or plain chunking based on language hint
/// and source_path extension. Mirrors the heuristic in
/// `semdoc.rs::detect_language`: `language == "markdown"` OR path ends in
/// `.md` → markdown path. Reads `SEMDOC_CHUNK_SIZE` /
/// `SEMDOC_CHUNK_OVERLAP` env vars at call time so callers can tune
/// chunk size without recompiling or restarting (e.g. larger chunks for
/// recall-heavy workloads, smaller for precision-heavy).
pub fn chunk_for(
    text: &str,
    parent_doc_id: &str,
    language: Option<&str>,
    source_path: &str,
) -> Vec<Chunk> {
    let chunk_size = resolve_chunk_size();
    let overlap = resolve_chunk_overlap(chunk_size);
    let is_md = language.map(|l| l == "markdown").unwrap_or(false)
        || source_path.ends_with(".md")
        || source_path.ends_with(".markdown");
    if is_md {
        chunk_markdown_with(text, parent_doc_id, chunk_size, overlap)
    } else {
        chunk_plain_with(text, parent_doc_id, chunk_size, overlap)
    }
}

// ----------------------------------------------------------------------------
// internals
// ----------------------------------------------------------------------------

/// Split markdown into sections on `##`, `###`, `####` ATX headers. Each
/// section's text starts with the header line itself (so chunk consumers
/// know which section they're in). Leading preamble before the first
/// header becomes its own section. Setext headers (`===` / `---` under
/// text) are NOT split — rare in kernel docs and ambiguous to detect.
fn split_markdown_by_headers(text: &str) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        // Match `^#{2,4}\s` — exactly 2-4 leading hashes followed by space.
        // `# title` (h1) and `##### deeper` (h5+) don't split.
        let is_split_header = line.starts_with("## ")
            || line.starts_with("### ")
            || line.starts_with("#### ");

        if is_split_header && !current.trim().is_empty() {
            sections.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        sections.push(current);
    }

    // Edge case: input was empty or all whitespace.
    if sections.is_empty() {
        return Vec::new();
    }
    sections
}

/// Recursive character split, LangChain `RecursiveCharacterTextSplitter`
/// style. Greedily accumulate text separated by the highest-priority
/// separator (`\n\n` first, then `\n`, then space) until adding another
/// piece would exceed `chunk_size`; flush the buffer and start the next
/// chunk with `overlap` chars of the previous chunk's tail.
///
/// `chunk_size` is in Unicode chars, not bytes — UTF-8 safe on Chinese
/// text. Token-budgeted splitting would require a tokenizer dependency;
/// char budget is good enough for our use case (bge-m3 truncates at 8192
/// tokens so 512 chars ≈ 200-400 tokens is well within the model's
/// capacity).
fn recursive_char_split(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if chunk_size == 0 {
        return Vec::new();
    }
    let total_chars = text.chars().count();
    if total_chars <= chunk_size {
        return vec![text.to_string()];
    }

    // Split on `\n\n` (highest priority) — this is what kernel markdown
    // actually uses between paragraphs. If even a single paragraph is
    // bigger than chunk_size, hard-split it.
    let pieces: Vec<&str> = text.split("\n\n").collect();
    let mut out: Vec<String> = Vec::new();
    let mut buffer = String::new();

    for piece in pieces {
        let piece_chars = piece.chars().count();

        // If the piece alone is bigger than chunk_size, hard-split it.
        if piece_chars > chunk_size {
            // Flush current buffer first.
            if !buffer.trim().is_empty() {
                out.push(std::mem::take(&mut buffer));
            }
            // Then hard-split the oversized piece, with overlap chained
            // from the last flushed chunk's tail.
            let seed_overlap: String = if overlap > 0 {
                out.last()
                    .map(|s: &String| s.chars().rev().take(overlap).collect::<Vec<_>>())
                    .unwrap_or_default()
                    .into_iter()
                    .rev()
                    .collect()
            } else {
                String::new()
            };
            let mut prefixed = seed_overlap.clone();
            prefixed.push_str(piece);
            hard_split_chars(&prefixed, chunk_size, overlap, &mut out);
            // After hard-split, the buffer's overlap tail becomes the
            // natural seed for the next piece. Reset buffer to overlap
            // tail of last out chunk.
            buffer = if overlap > 0 {
                out.last()
                    .map(|s: &String| s.chars().rev().take(overlap).collect::<Vec<_>>())
                    .unwrap_or_default()
                    .into_iter()
                    .rev()
                    .collect()
            } else {
                String::new()
            };
            continue;
        }

        // Piece fits — does adding it to buffer overflow?
        let candidate_len = buffer.chars().count() + 2 + piece_chars; // +2 for "\n\n" join
        if candidate_len > chunk_size && !buffer.trim().is_empty() {
            // Flush buffer, seed next with overlap tail.
            let tail: String = if overlap > 0 {
                buffer
                    .chars()
                    .rev()
                    .take(overlap)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            } else {
                String::new()
            };
            out.push(std::mem::take(&mut buffer));
            buffer = tail;
            if !buffer.is_empty() {
                buffer.push_str("\n\n");
            }
            buffer.push_str(piece);
        } else {
            if !buffer.is_empty() {
                buffer.push_str("\n\n");
            }
            buffer.push_str(piece);
        }
    }
    if !buffer.trim().is_empty() {
        out.push(buffer);
    }

    out.retain(|s| !s.trim().is_empty());
    out
}

/// Hard split by chunk_size chars with overlap. Each chunk is at most
/// `chunk_size` chars (including any overlap prefix carried in from the
/// previous chunk), advancing by `chunk_size - overlap` per chunk so
/// adjacent chunks share `overlap` chars of context.
fn hard_split_chars(text: &str, chunk_size: usize, overlap: usize, out: &mut Vec<String>) {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return;
    }
    // When overlap > 0, each emitted chunk's effective content window
    // shrinks to `chunk_size - overlap` so the overlap tail of chunk N
    // becomes the head of chunk N+1 while staying within chunk_size total.
    let window = if overlap > 0 && overlap < chunk_size {
        chunk_size - overlap
    } else {
        chunk_size
    };
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let piece: String = chars[start..end].iter().collect();
        if !piece.trim().is_empty() {
            out.push(piece);
        }
        if end >= chars.len() {
            break;
        }
        // Advance by window, clamped to >=1 so we always make progress.
        let step = window.max(1);
        start += step;
    }
}

fn finalize_chunks(pieces: Vec<(String, String)>, parent_doc_id: &str) -> Vec<Chunk> {
    pieces
        .into_iter()
        .enumerate()
        .map(|(i, (text, _))| Chunk {
            chunk_id: Chunk::compute_id(parent_doc_id, i as u32, &text),
            text,
            parent_doc_id: parent_doc_id.to_string(),
            chunk_index: i as u32,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_id_is_deterministic() {
        let a = Chunk::compute_id("parent", 0, "hello");
        let b = Chunk::compute_id("parent", 0, "hello");
        assert_eq!(a, b);
    }

    #[test]
    fn chunk_id_changes_with_index() {
        let a = Chunk::compute_id("parent", 0, "hello");
        let b = Chunk::compute_id("parent", 1, "hello");
        assert_ne!(a, b);
    }

    #[test]
    fn chunk_id_changes_with_parent() {
        let a = Chunk::compute_id("parent-a", 0, "hello");
        let b = Chunk::compute_id("parent-b", 0, "hello");
        assert_ne!(a, b);
    }

    #[test]
    fn short_text_stays_one_chunk_plain() {
        let chunks = chunk_plain("hello world", "p1");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        let chunks = chunk_plain("", "p1");
        assert!(chunks.is_empty());
    }

    #[test]
    fn whitespace_only_produces_no_chunks() {
        let chunks = chunk_plain("   \n\n  \t  ", "p1");
        assert!(chunks.is_empty());
    }

    #[test]
    fn long_text_splits_with_overlap() {
        // 2000 chars of 'a' should produce ≥3 chunks of 512 with 50 overlap.
        let text: String = "a".repeat(2000);
        let chunks = chunk_plain(&text, "p1");
        assert!(chunks.len() >= 3, "expected >= 3 chunks, got {}", chunks.len());
        for c in &chunks {
            assert!(c.text.chars().count() <= 512);
        }
    }

    #[test]
    fn markdown_sections_split_on_h2() {
        // Sections shorter than the coalesce threshold get merged into the
        // next section, so each section body here must exceed it (~170 chars)
        // to stand alone.
        let fill = |s: &str| format!("{} {}", s, "word ".repeat(40));
        let md = format!(
            "# Title\n\n{}\n\n## Section A\n\n{}\n\n## Section B\n\n{}",
            fill("preamble"),
            fill("content A"),
            fill("content B")
        );
        let chunks = chunk_markdown(&md, "p1");
        // Title + preamble → 1 chunk; Section A → 1; Section B → 1.
        assert_eq!(chunks.len(), 3, "got chunks: {:?}", chunks);
        assert!(chunks[0].text.contains("preamble"));
        assert!(chunks[0].text.contains("# Title"));
        assert!(chunks[1].text.contains("Section A"));
        assert!(chunks[2].text.contains("Section B"));
    }

    #[test]
    fn markdown_short_sections_coalesce_into_next() {
        // Short sections are merged forward rather than emitted as tiny
        // chunks (the "1-line header is its own chunk" failure mode).
        let md = "# Title\n\npreamble\n\n## A\n\nshort A\n\n## B\n\nshort B";
        let chunks = chunk_markdown(md, "p1");
        assert_eq!(chunks.len(), 1, "got chunks: {:?}", chunks);
        assert!(chunks[0].text.contains("short A"));
        assert!(chunks[0].text.contains("short B"));
    }

    #[test]
    fn markdown_h1_is_not_a_split_point() {
        let md = "# A\n\n# B\n\nbody";
        let chunks = chunk_markdown(md, "p1");
        // Both h1s stay together — they're not split points.
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("# A"));
        assert!(chunks[0].text.contains("# B"));
    }

    #[test]
    fn long_markdown_section_gets_recursive_split() {
        // Build a section with a 1500-char body that should split into
        // ~3 chunks.
        let body: String = "x".repeat(1500);
        let md = format!("## Long\n\n{body}");
        let chunks = chunk_markdown(&md, "p1");
        assert!(chunks.len() >= 2, "expected >= 2 chunks, got {}", chunks.len());
        // Each chunk's text should include the header (overlap behavior).
    }

    #[test]
    fn chunk_for_dispatches_on_extension() {
        let plain_chunks = chunk_for("hello world", "p1", None, "/tmp/foo.txt");
        assert_eq!(plain_chunks.len(), 1);

        let md_chunks = chunk_for("# H\n\nbody", "p1", None, "/tmp/foo.md");
        assert_eq!(md_chunks.len(), 1);
    }

    #[test]
    fn chunk_for_dispatches_on_language_hint() {
        let chunks = chunk_for("hello world", "p1", Some("markdown"), "/tmp/foo.txt");
        // markdown path with plain text → 1 chunk (no headers to split on).
        assert_eq!(chunks.len(), 1);
    }
}
