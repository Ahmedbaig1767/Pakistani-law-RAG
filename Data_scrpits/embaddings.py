"""
============================================================
  embeddings.py
  Phase 4 — Embedding & Vector Storage

  Loads all chunks, embeds them using sentence-transformers,
  and stores them in ChromaDB for retrieval.

  Model : sentence-transformers/all-MiniLM-L6-v2
            - Free, local, no API key
            - 384 dimensions
            - Fast and accurate for legal Q&A

  Vector DB : ChromaDB
            - Local, no signup
            - Metadata filtering by doc_name
            - Persists to disk automatically

  Install:
    pip install sentence-transformers chromadb

  Output:
    data/vectorstore/   ← ChromaDB persisted to disk
============================================================
"""

import json
import os
import time

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ─────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────

CHUNKS_PATH   = "data/chunks/all_chunks.json"
VECTORSTORE   = "data/vectorstore"
COLLECTION    = "legal_docs"
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE    = 64   # embed 64 chunks at a time


# ─────────────────────────────────────────────
# STEP 1 — Load Chunks
# ─────────────────────────────────────────────

def load_chunks(path):
    """
    Loads all chunks from all_chunks.json.
    Returns list of chunk dicts.
    """
    print(f"\n{'='*55}")
    print(f"  📂 LOADING CHUNKS")
    print(f"{'='*55}")

    if not os.path.exists(path):
        print(f"  ❌ File not found: {path}")
        print(f"     Run chunker.py first!")
        return []

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  ✅ Loaded {len(chunks)} chunks")

    # Quick breakdown by document
    for doc in ["constitution", "ppc", "crpc"]:
        count = sum(1 for c in chunks if c["doc_name"] == doc)
        print(f"     {doc:<20} → {count} chunks")

    return chunks


# ─────────────────────────────────────────────
# STEP 2 — Load Embedding Model
# ─────────────────────────────────────────────

def load_model(model_name):
    """
    Loads sentence-transformers model locally.
    Downloads once, cached after that.
    """
    print(f"\n{'='*55}")
    print(f"  🤖 LOADING EMBEDDING MODEL")
    print(f"{'='*55}")
    print(f"  Model : {model_name}")
    print(f"  (Downloads on first run, cached after)")

    model = SentenceTransformer(model_name)

    print(f"  ✅ Model loaded")
    print(f"  Embedding dimensions : {model.get_sentence_embedding_dimension()}")

    return model


# ─────────────────────────────────────────────
# STEP 3 — Setup ChromaDB
# ─────────────────────────────────────────────

def setup_chromadb(persist_dir, collection_name):
    """
    Creates or loads a ChromaDB collection.
    Persists to disk so embeddings survive restarts.
    """
    print(f"\n{'='*55}")
    print(f"  🗄️  SETTING UP CHROMADB")
    print(f"{'='*55}")

    os.makedirs(persist_dir, exist_ok=True)

    # Initialize ChromaDB with disk persistence
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if re-running
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"  ⚠️  Collection '{collection_name}' exists — deleting and rebuilding")
        client.delete_collection(collection_name)

    # Create fresh collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text
    )

    print(f"  ✅ Collection '{collection_name}' ready")
    print(f"  📁 Persisted at: {persist_dir}")

    return client, collection


# ─────────────────────────────────────────────
# STEP 4 — Embed & Store
# ─────────────────────────────────────────────

def embed_and_store(chunks, model, collection, batch_size=64):
    print(f"\n{'='*55}")
    print(f"  ⚙️  EMBEDDING & STORING CHUNKS")
    print(f"{'='*55}")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Total batches: {(len(chunks) + batch_size - 1) // batch_size}")

    total_stored = 0
    start_time   = time.time()

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        texts = [c["text"] for c in batch]

        # ✅ FIXED HERE
        embeddings = model.encode(
            texts,
            show_progress_bar=False
        ).tolist()

        ids       = [c["chunk_id"] for c in batch]
        documents = texts

        metadatas = [
            {
                "doc_name":    c["doc_name"],
                "section_num": str(c["section_num"]),
                "title":       c["title"],
                "word_count":  c["word_count"]
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        total_stored += len(batch)
        batch_num = (batch_start // batch_size) + 1
        elapsed = time.time() - start_time

        print(f"  Batch {batch_num:>3} | Stored {total_stored:>4}/{len(chunks)} chunks | {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n  ✅ All chunks embedded and stored!")
    print(f"  ⏱️  Total time : {elapsed:.1f} seconds")
    print(f"  📦 Total stored: {total_stored}")

    return total_stored


# ─────────────────────────────────────────────
# STEP 5 — Verify Storage
# ─────────────────────────────────────────────

def verify_storage(collection):
    """
    Quick sanity check — runs a test query to confirm
    embeddings are working correctly.
    """
    print(f"\n{'='*55}")
    print(f"  🔍 VERIFYING WITH TEST QUERY")
    print(f"{'='*55}")

    test_queries = [
        ("punishment for murder",      "ppc"),
        ("arrest without warrant",     "crpc"),
        ("fundamental rights",         "constitution"),
    ]

    for query_text, expected_doc in test_queries:
        results = collection.query(
            query_texts = [query_text],
            n_results   = 1
        )

        top_id    = results["ids"][0][0]
        top_meta  = results["metadatas"][0][0]
        top_doc   = top_meta["doc_name"]
        top_title = top_meta["title"]
        distance  = results["distances"][0][0]

        match = "✅" if top_doc == expected_doc else "⚠️ "
        print(f"\n  Query   : '{query_text}'")
        print(f"  Result  : {match} [{top_doc}] {top_title}")
        print(f"  Chunk ID: {top_id}  |  Distance: {distance:.4f}")


# ─────────────────────────────────────────────
# STEP 6 — Print Final Stats
# ─────────────────────────────────────────────

def print_stats(collection):
    """
    Prints final collection stats.
    """
    total = collection.count()

    print(f"\n{'='*55}")
    print(f"  📊 FINAL STATS")
    print(f"{'='*55}")
    print(f"  Collection    : {COLLECTION}")
    print(f"  Total vectors : {total}")
    print(f"  Stored at     : {VECTORSTORE}/")
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Dimensions    : 384")
    print(f"  Similarity    : Cosine")
    print(f"\n  ✅ Ready for retrieval in chatbot!")
    print(f"{'='*55}")


# ─────────────────────────────────────────────
# Master Pipeline
# ─────────────────────────────────────────────

def build_vectorstore():
    """
    Full pipeline:
      1. Load chunks
      2. Load model
      3. Setup ChromaDB
      4. Embed + store
      5. Verify
      6. Stats
    """
    print(f"\n{'='*55}")
    print(f"  🚀 EMBEDDINGS PIPELINE STARTING")
    print(f"{'='*55}")

    # Step 1 — Load chunks
    chunks = load_chunks(CHUNKS_PATH)
    if not chunks:
        return

    # Step 2 — Load model
    model = load_model(MODEL_NAME)

    # Step 3 — Setup ChromaDB
    client, collection = setup_chromadb(VECTORSTORE, COLLECTION)

    # Step 4 — Embed and store
    embed_and_store(chunks, model, collection, batch_size=BATCH_SIZE)

    # Step 5 — Verify with test queries
    verify_storage(collection)

    # Step 6 — Final stats
    print_stats(collection)


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    build_vectorstore()