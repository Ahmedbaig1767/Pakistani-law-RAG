"""
============================================================
  chunker.py - COMPLETE FIXED VERSION
============================================================
"""

import json
import os
import re

INPUT_DIR = "data/normalized"
OUTPUT_DIR = "data/chunks"

# ─────────────────────────────────────────────
# ✅ FIXED: Core Section Splitter (handles duplicates)
# ─────────────────────────────────────────────
def split_by_sections(full_text, doc_name, pattern, label="Section"):
    chunks = []
    matches = list(pattern.finditer(full_text))
    MAX_WORDS = 400
    OVERLAP = 50

    # ✅ DUPLICATE PREVENTION
    seen_sections = set()

    print(f"  Found {len(matches)} section headers")

    for i, match in enumerate(matches):
        section_num = match.group(1).strip()
        title = match.group(2).strip()

        # ✅ SKIP DUPLICATES
        if section_num in seen_sections:
            print(f"  ⏭️  Skipping duplicate: {doc_name}_{section_num}")
            continue
        seen_sections.add(section_num)

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()

        word_count = len(text.split())
        if word_count < 20:
            continue

        # ✅ parent_id for normal chunks (no sub-chunking)
        parent_id = f"{doc_name}_{section_num}"

        if word_count > MAX_WORDS:
            words = text.split()
            start_idx = 0
            sub_i = 1
            while start_idx < len(words):
                end_idx = start_idx + MAX_WORDS
                sub_text = " ".join(words[start_idx:end_idx])
                chunks.append({
                    "doc_name":   doc_name,
                    "chunk_id":   f"{doc_name}_{section_num}_{sub_i}",
                    "parent_id":  parent_id,          # ✅ links back to section
                    "section_num": section_num,
                    "title":      f"{label} {section_num} - {title} (part {sub_i})",
                    "text":       sub_text,
                    "word_count": len(sub_text.split())
                })
                start_idx += MAX_WORDS - OVERLAP
                sub_i += 1
            continue

        chunks.append({
            "doc_name":    doc_name,
            "chunk_id":    f"{doc_name}_{section_num}",
            "parent_id":   parent_id,                 # ✅ same as chunk_id for root chunks
            "section_num": section_num,
            "title":       f"{label} {section_num} - {title}",
            "text":        text,
            "word_count":  word_count
        })

    # ✅ VERIFY UNIQUENESS
    chunk_ids = [c["chunk_id"] for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"❌ Duplicate chunk_ids in {doc_name}")

    print(f"  ✅ {len(chunks)} unique chunks created")
    return chunks

# ─────────────────────────────────────────────
# ✅ CONSTITUTION CHUNKER
# ─────────────────────────────────────────────
def chunk_constitution(pages):
    full_text = "\n".join(p["text"] for p in pages)
    # ✅ IMPROVED REGEX: Article boundaries
    pattern = re.compile(r'\n(\d{1,3})\.\s+([a-z][^\n]{3,100})\n', re.MULTILINE)
    return split_by_sections(full_text, "constitution", pattern, label="Article")

# ─────────────────────────────────────────────
# ✅ PPC CHUNKER  
# ─────────────────────────────────────────────
def chunk_ppc(pages):
    full_text = "\n".join(p["text"] for p in pages)
    pattern = re.compile(r'\n(\d{2,3})\.\s+([a-z][^\n]{3,100})\n', re.MULTILINE)
    return split_by_sections(full_text, "ppc", pattern, label="Section")

# ─────────────────────────────────────────────
# ✅ CRPC CHUNKER
# ─────────────────────────────────────────────
def chunk_crpc(pages):
    full_text = "\n".join(p["text"] for p in pages)
    pattern = re.compile(r'\n(\d{1,3})\.\s+([a-z][^\n]{3,100})\n', re.MULTILINE)
    return split_by_sections(full_text, "crpc", pattern, label="Section")

# ─────────────────────────────────────────────
# ✅ MASTER PIPELINE
# ─────────────────────────────────────────────
def chunk_document(doc_name):
    input_path = os.path.join(INPUT_DIR, f"{doc_name}_normalized.json")
    print(f"\n{'='*55}")
    print(f"  ✂️  CHUNKING: {doc_name.upper()}")
    print(f"{'='*55}")

    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        return []

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"  Loaded {len(pages)} normalized pages")

    if doc_name == "constitution":
        chunks = chunk_constitution(pages)
    elif doc_name == "ppc":
        chunks = chunk_ppc(pages)
    elif doc_name == "crpc":
        chunks = chunk_crpc(pages)
    else:
        chunks = []

    if not chunks:
        print(f"  ❌ No sections found!")
        return []

    word_counts = [c["word_count"] for c in chunks]
    print(f"\n  ✅ Chunking complete!")
    print(f"  Total chunks: {len(chunks)} | Avg: {sum(word_counts)/len(word_counts):.0f} words")
    
    # Sample
    for chunk in chunks[:2]:
        print(f"  📌 {chunk['chunk_id']}: {chunk['title'][:60]}...")
    
    return chunks

def chunk_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    documents = ["constitution", "ppc", "crpc"]
    all_chunks = []
    
    for doc_name in documents:
        chunks = chunk_document(doc_name)
        if chunks:
            output_path = os.path.join(OUTPUT_DIR, f"{doc_name}_chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"  💾 {output_path}")
            all_chunks.extend(chunks)
    
    # ✅ GLOBAL UNIQUENESS CHECK
    all_ids = [c["chunk_id"] for c in all_chunks]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("❌ GLOBAL DUPLICATES FOUND!")
    
    combined_path = os.path.join(OUTPUT_DIR, "all_chunks.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 {len(all_chunks)} UNIQUE CHUNKS READY!")
    print(f"📁 data/chunks/all_chunks.json")
    return all_chunks

if __name__ == "__main__":
    chunk_all()