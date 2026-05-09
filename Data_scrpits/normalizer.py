

import json
import os
import re


INPUT_DIR  = "data/processed"
OUTPUT_DIR = "data/normalized"


# ─────────────────────────────────────────────
# STEP 1 — TEXT NORMALIZATION
# ─────────────────────────────────────────────

def normalize_text(text):
    """
    Normalize text for better embedding quality.

    Operations:
      1. Lowercase         → consistent text for embeddings
      2. Keep dots + dashes → section numbers stay intact
                              e.g. "379." stays as "379."
      3. Remove other special chars
      4. Normalize whitespace
    """

    # Step 1 — Lowercase everything
    text = text.lower()

    # Step 2 — Remove special characters BUT keep:
    #   \w  = letters and numbers
    #   \s  = spaces and newlines
    #   \.  = dots        (keeps "379." "Art. 25")
    #   \-  = hyphens     (keeps compound words)
    #   \(  = open paren  (keeps legal clauses like "(1)")
    #   \)  = close paren (keeps legal clauses like "(1)")
    text = re.sub(r'[^\w\s.\-\(\)]', ' ', text)

    # Step 3 — Normalize whitespace (multiple spaces → single space)
    # But preserve newlines so section patterns still work
    text = re.sub(r'[ \t]+', ' ', text)

    # Step 4 — Remove more than 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def normalize_pages(pages):
    """
    Apply normalization to all pages.
    """
    for page in pages:
        page["text"] = normalize_text(page["text"])

    print(f"  [Normalization] Applied to {len(pages)} pages")

    # Quick sanity check — show a sample
    if pages:
        sample = pages[0]["text"][:150]
        print(f"  [Sample]        {sample}...")

    return pages


# ─────────────────────────────────────────────
# STEP 2 — FEATURE SCALING
# ─────────────────────────────────────────────

def min_max_scale(values):
    """
    Scale values between 0 and 1 using Min-Max Scaling.

    Formula: (value - min) / (max - min)
    """
    min_val = min(values)
    max_val = max(values)

    # Avoid division by zero if all values are same
    if max_val == min_val:
        return [0.5 for _ in values]

    return [(v - min_val) / (max_val - min_val) for v in values]


def scale_word_counts(pages):
    """
    Scale word_count feature for all pages using Min-Max scaling.
    Adds a new field word_count_scaled (value between 0 and 1).
    """
    word_counts = [page["word_count"] for page in pages]
    scaled      = min_max_scale(word_counts)

    for page, s in zip(pages, scaled):
        page["word_count_scaled"] = round(s, 4)

    print(f"  [Scaling]       Word counts scaled for {len(pages)} pages")
    print(f"  [Range]         Min={min(word_counts)} words | Max={max(word_counts)} words")

    return pages


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def normalize_document(doc_name):
    """
    Runs full normalization pipeline on one document.
    """
    input_path = os.path.join(INPUT_DIR, f"{doc_name}_clean.json")

    print(f"\n{'='*55}")
    print(f"  🔧 NORMALIZING: {doc_name.upper()}")
    print(f"{'='*55}")

    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        print(f"     Run preprocessor.py first!")
        return []

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"  Loaded {len(pages)} cleaned pages")

    # Step 1 — Normalize text
    pages = normalize_pages(pages)

    # Step 2 — Scale word counts
    pages = scale_word_counts(pages)

    return pages


def normalize_all():
    """
    Runs normalization on all 3 legal documents.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    documents = ["constitution", "ppc", "crpc"]

    print("=" * 55)
    print("  🔧 STARTING NORMALIZATION PIPELINE")
    print("=" * 55)

    for doc_name in documents:
        pages = normalize_document(doc_name)

        if not pages:
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{doc_name}_normalized.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)

        print(f"  💾 Saved to: {output_path}")

    print(f"\n{'='*55}")
    print(f"  ✅ Normalization complete!")
    print(f"  📁 Files saved in: {OUTPUT_DIR}/")
    print(f"{'='*55}")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    normalize_all()