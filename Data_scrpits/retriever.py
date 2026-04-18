"""
============================================================
  retriever.py
  Phase 5 — Hybrid Retrieval (Semantic + Keyword)

  Requirement 1: Hybrid Search
    - Semantic Search  → ChromaDB + sentence-transformers
    - Keyword Search   → BM25 (handles legal jargon exactly)
    - Combined Score   → weighted fusion of both

  Requirement 2: Hierarchical Chunking
    - Detects sub-chunks (e.g. constitution_91_1, _2, _3)
    - Groups them by parent_id
    - Merges sibling parts for complete legal context

  Output per query:
    [
      {
        "chunk_id":    "constitution_91",
        "parent_id":   "constitution_91",
        "doc_name":    "constitution",
        "title":       "Article 91 - the cabinet",
        "text":        "full merged text...",
        "word_count":  400,
        "score":       0.87,          ← hybrid score
        "semantic_score": 0.91,
        "keyword_score":  0.74,
        "source_snippet": "first 300 chars...",   ← for Side-by-Side UI
      },
      ...
    ]

  Install:
    pip install rank-bm25 sentence-transformers chromadb

============================================================
"""

import json
import os
import math
import re
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CHUNKS_PATH  = "data/chunks/all_chunks.json"
VECTORSTORE  = "data/vectorstore"
COLLECTION   = "legal_docs"
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"

# Hybrid Search Weights
# Tune these if results are off:
#   More weight on SEMANTIC → better for natural language queries
#   More weight on KEYWORD  → better for specific legal terms/sections
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT  = 0.4

# How many candidates to fetch from each search before fusion
TOP_K_SEMANTIC  = 10
TOP_K_KEYWORD   = 10

# Final results to return
TOP_K_FINAL     = 5


# ─────────────────────────────────────────────
# STEP 1 — Load Everything
# ─────────────────────────────────────────────

def load_resources():
    """
    Loads:
      - All chunks from disk (for BM25)
      - Embedding model (for semantic search)
      - ChromaDB collection (for vector search)
    """
    print("  📂 Loading chunks for BM25...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    print(f"     {len(all_chunks)} chunks loaded")

    print("  🤖 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"     Model ready")

    print("  🗄️  Connecting to ChromaDB...")
    client     = chromadb.PersistentClient(path=VECTORSTORE)
    collection = client.get_collection(COLLECTION)
    print(f"     {collection.count()} vectors ready")

    return all_chunks, model, collection


# ─────────────────────────────────────────────
# STEP 2 — Build BM25 Index
# ─────────────────────────────────────────────

def build_bm25_index(chunks):
    """
    Builds a BM25 index over all chunk texts.
    BM25 is great for exact legal terms like:
      - "section 302"
      - "cognizable offence"
      - "writ of habeas corpus"
      - "prima facie"
    """
    # Tokenize each chunk text
    tokenized = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized)
    print(f"  📑 BM25 index built over {len(chunks)} chunks")
    return bm25


# ─────────────────────────────────────────────
# STEP 3 — Semantic Search
# ─────────────────────────────────────────────

def semantic_search(query, model, collection, top_k=10):
    """
    Embeds the query and searches ChromaDB for
    semantically similar chunks.

    Returns list of:
      { chunk_id, score (0-1, higher=better), metadata, text }
    """
    results = collection.query(
        query_texts = [query],
        n_results   = top_k
    )

    semantic_results = []
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        distance = results["distances"][0][i]      # cosine distance (lower = better)
        score    = 1 - distance                    # convert to similarity (higher = better)
        metadata = results["metadatas"][0][i]
        text     = results["documents"][0][i]

        semantic_results.append({
            "chunk_id": chunk_id,
            "score":    score,
            "metadata": metadata,
            "text":     text
        })

    return semantic_results


# ─────────────────────────────────────────────
# STEP 4 — Keyword Search (BM25)
# ─────────────────────────────────────────────

def keyword_search(query, bm25, chunks, top_k=10):
    """
    Searches using BM25 keyword matching.
    Best for exact legal section names, article numbers,
    and domain-specific legal jargon.

    Returns list of:
      { chunk_id, score (normalized 0-1), metadata, text }
    """
    tokenized_query = query.lower().split()
    scores          = bm25.get_scores(tokenized_query)

    # Get top_k indices sorted by score descending
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # Normalize BM25 scores to 0-1
    max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1

    keyword_results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue

        chunk    = chunks[idx]
        norm_score = scores[idx] / max_score      # normalize to 0-1

        keyword_results.append({
            "chunk_id": chunk["chunk_id"],
            "score":    norm_score,
            "metadata": {
                "doc_name":    chunk["doc_name"],
                "section_num": str(chunk["section_num"]),
                "title":       chunk["title"],
                "word_count":  chunk["word_count"]
            },
            "text": chunk["text"]
        })

    return keyword_results


# ─────────────────────────────────────────────
# STEP 5 — Hybrid Fusion
# ─────────────────────────────────────────────

def hybrid_fusion(semantic_results, keyword_results,
                  semantic_weight=0.6, keyword_weight=0.4):
    """
    Combines semantic and keyword scores using
    weighted score fusion.

    Formula:
      hybrid_score = (semantic_weight * semantic_score)
                   + (keyword_weight  * keyword_score)

    Chunks that appear in BOTH searches get a boosted score.
    Chunks that appear in only one still get partial credit.
    """
    scores = defaultdict(lambda: {
        "semantic_score": 0.0,
        "keyword_score":  0.0,
        "metadata":       None,
        "text":           ""
    })

    # Add semantic scores
    for r in semantic_results:
        cid = r["chunk_id"]
        scores[cid]["semantic_score"] = r["score"]
        scores[cid]["metadata"]       = r["metadata"]
        scores[cid]["text"]           = r["text"]

    # Add keyword scores
    for r in keyword_results:
        cid = r["chunk_id"]
        scores[cid]["keyword_score"] = r["score"]
        if scores[cid]["metadata"] is None:
            scores[cid]["metadata"] = r["metadata"]
            scores[cid]["text"]     = r["text"]

    # Compute hybrid score
    fused = []
    for chunk_id, data in scores.items():
        hybrid_score = (
            semantic_weight * data["semantic_score"] +
            keyword_weight  * data["keyword_score"]
        )
        fused.append({
            "chunk_id":       chunk_id,
            "hybrid_score":   hybrid_score,
            "semantic_score": data["semantic_score"],
            "keyword_score":  data["keyword_score"],
            "metadata":       data["metadata"],
            "text":           data["text"]
        })

    # Sort by hybrid score descending
    fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return fused


# ─────────────────────────────────────────────
# STEP 6 — Hierarchical Merging
# ─────────────────────────────────────────────

def merge_hierarchical(fused_results, all_chunks, top_k=5):
    """
    Requirement 2: Hierarchical Chunking support.

    If sub-chunks of the same parent section are retrieved
    (e.g. constitution_91_1 AND constitution_91_2),
    merge them into one complete response so the LLM
    gets the full legal context.

    Also fetches the parent chunk text if only a sub-chunk
    was retrieved, to provide full section context.
    """
    # Build lookup: chunk_id → chunk
    chunk_lookup = {c["chunk_id"]: c for c in all_chunks}

    # Group results by parent_id
    parent_groups = defaultdict(list)
    for result in fused_results:
        chunk_id  = result["chunk_id"]
        chunk_data = chunk_lookup.get(chunk_id, {})
        parent_id  = chunk_data.get("parent_id", chunk_id)
        parent_groups[parent_id].append(result)

    # Merge groups
    merged = []
    seen_parents = set()

    for result in fused_results:
        chunk_id   = result["chunk_id"]
        chunk_data = chunk_lookup.get(chunk_id, {})
        parent_id  = chunk_data.get("parent_id", chunk_id)

        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        group = parent_groups[parent_id]

        if len(group) > 1:
            # Multiple sub-chunks of same section retrieved
            # Sort by chunk_id to get correct order (part_1, part_2...)
            group_sorted = sorted(group, key=lambda x: x["chunk_id"])

            # Merge texts in order
            merged_text = " ".join(g["text"] for g in group_sorted)

            # Use highest score from group
            best_score    = max(g["hybrid_score"]   for g in group_sorted)
            best_semantic = max(g["semantic_score"]  for g in group_sorted)
            best_keyword  = max(g["keyword_score"]   for g in group_sorted)

            metadata = group_sorted[0]["metadata"]

            merged.append({
                "chunk_id":       parent_id,
                "parent_id":      parent_id,
                "doc_name":       metadata.get("doc_name", ""),
                "section_num":    metadata.get("section_num", ""),
                "title":          metadata.get("title", "").replace(" (part 1)", ""),
                "text":           merged_text,
                "word_count":     len(merged_text.split()),
                "hybrid_score":   round(best_score,    4),
                "semantic_score": round(best_semantic, 4),
                "keyword_score":  round(best_keyword,  4),
                "source_snippet": merged_text[:300],   # for Side-by-Side UI
                "parts_merged":   len(group)
            })
        else:
            # Single chunk
            r        = group[0]
            metadata = r["metadata"]
            text     = r["text"]

            merged.append({
                "chunk_id":       chunk_id,
                "parent_id":      parent_id,
                "doc_name":       metadata.get("doc_name", ""),
                "section_num":    metadata.get("section_num", ""),
                "title":          metadata.get("title", ""),
                "text":           text,
                "word_count":     len(text.split()),
                "hybrid_score":   round(r["hybrid_score"],   4),
                "semantic_score": round(r["semantic_score"], 4),
                "keyword_score":  round(r["keyword_score"],  4),
                "source_snippet": text[:300],              # for Side-by-Side UI
                "parts_merged":   1
            })

    # Re-sort after merging and return top_k
    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged[:top_k]


# ─────────────────────────────────────────────
# STEP 7 — Main Retrieve Function
# ─────────────────────────────────────────────

def retrieve(query, model, collection, bm25, all_chunks,
             top_k=TOP_K_FINAL,
             semantic_weight=SEMANTIC_WEIGHT,
             keyword_weight=KEYWORD_WEIGHT,
             doc_filter=None):
    """
    Full hybrid retrieval pipeline for one query.

    Parameters:
        query          (str)  : User's question
        model                 : SentenceTransformer model
        collection            : ChromaDB collection
        bm25                  : BM25 index
        all_chunks     (list) : All chunks from JSON
        top_k          (int)  : Number of results to return
        semantic_weight(float): Weight for semantic score
        keyword_weight (float): Weight for keyword score
        doc_filter     (str)  : Optional — "constitution", "ppc", "crpc"
                                Filters results to one document only

    Returns:
        list of merged result dicts (sorted by hybrid score)
    """

    # Optional document filter
    filtered_chunks = all_chunks
    if doc_filter:
        filtered_chunks = [
            c for c in all_chunks
            if c["doc_name"] == doc_filter
        ]
        # Rebuild BM25 on filtered set
        filtered_bm25 = build_bm25_index(filtered_chunks)
    else:
        filtered_bm25 = bm25

    # Step 1 — Semantic search
    semantic_results = semantic_search(
        query, model, collection, top_k=TOP_K_SEMANTIC
    )

    # Apply doc_filter to semantic results
    if doc_filter:
        semantic_results = [
            r for r in semantic_results
            if r["metadata"].get("doc_name") == doc_filter
        ]

    # Step 2 — Keyword search
    keyword_results = keyword_search(
        query, filtered_bm25, filtered_chunks, top_k=TOP_K_KEYWORD
    )

    # Step 3 — Hybrid fusion
    fused = hybrid_fusion(
        semantic_results, keyword_results,
        semantic_weight, keyword_weight
    )

    # Step 4 — Hierarchical merging
    final = merge_hierarchical(fused, filtered_chunks, top_k=top_k)

    return final


# ─────────────────────────────────────────────
# STEP 8 — Pretty Print Results
# ─────────────────────────────────────────────

def print_results(query, results):
    """
    Prints retrieval results clearly for debugging.
    Shows scores, doc source, and text preview.
    """
    print(f"\n{'='*60}")
    print(f"  🔍 QUERY: '{query}'")
    print(f"{'='*60}")

    if not results:
        print("  ❌ No results found")
        return

    for i, r in enumerate(results):
        parts_info = f" [{r['parts_merged']} parts merged]" if r["parts_merged"] > 1 else ""
        print(f"\n  Result {i+1}{parts_info}")
        print(f"  {'─'*55}")
        print(f"  Doc      : {r['doc_name'].upper()}")
        print(f"  Title    : {r['title']}")
        print(f"  Chunk ID : {r['chunk_id']}")
        print(f"  Score    : Hybrid={r['hybrid_score']:.4f} | "
              f"Semantic={r['semantic_score']:.4f} | "
              f"Keyword={r['keyword_score']:.4f}")
        print(f"  Preview  : {r['source_snippet'][:200]}...")


# ─────────────────────────────────────────────
# Interactive Test Mode
# ─────────────────────────────────────────────

def interactive_test(model, collection, bm25, all_chunks):
    """
    Interactive query testing loop.
    Lets you test queries without re-running embeddings.
    """
    print(f"\n{'='*60}")
    print(f"  🚀 HYBRID RETRIEVER — INTERACTIVE TEST")
    print(f"  Semantic weight : {SEMANTIC_WEIGHT}")
    print(f"  Keyword weight  : {KEYWORD_WEIGHT}")
    print(f"  Top K results   : {TOP_K_FINAL}")
    print(f"{'='*60}")
    print(f"  Commands:")
    print(f"    'quit'              → exit")
    print(f"    'filter:ppc'        → search only PPC")
    print(f"    'filter:crpc'       → search only CrPC")
    print(f"    'filter:constitution' → search only Constitution")
    print(f"    'filter:off'        → remove filter")
    print(f"{'='*60}\n")

    doc_filter = None

    while True:
        query = input("🔍 Query: ").strip()

        if not query:
            continue

        if query.lower() == "quit":
            print("👋 Exiting retriever.")
            break

        # Handle filter commands
        if query.lower().startswith("filter:"):
            val = query.split(":")[1].strip().lower()
            if val == "off":
                doc_filter = None
                print(f"  ✅ Filter removed — searching all documents\n")
            elif val in ["ppc", "crpc", "constitution"]:
                doc_filter = val
                print(f"  ✅ Filter set to: {doc_filter}\n")
            else:
                print(f"  ❌ Unknown filter. Use: ppc, crpc, constitution, or off\n")
            continue

        # Run retrieval
        results = retrieve(
            query      = query,
            model      = model,
            collection = collection,
            bm25       = bm25,
            all_chunks = all_chunks,
            doc_filter = doc_filter
        )

        print_results(query, results)
        print()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  ⚙️  INITIALIZING HYBRID RETRIEVER")
    print(f"{'='*60}")

    all_chunks, model, collection = load_resources()
    bm25 = build_bm25_index(all_chunks)

    print(f"\n  ✅ Retriever ready!\n")

    interactive_test(model, collection, bm25, all_chunks)