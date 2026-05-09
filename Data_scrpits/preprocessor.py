

# Import libraries
import json   # for reading/writing JSON files
import os     # for file handling
import re     # for text cleaning using patterns (regex)


INPUT_DIR  = "data/extracted"
OUTPUT_DIR = "data/processed"


# ─────────────────────────────────────────────
# STEP 1 — Handle Missing Values
# ─────────────────────────────────────────────

def handle_missing_values(pages):
    """
    Removes pages that have no useful text.
    """
    before = len(pages) # pages before cleaning
    cleaned = []

    for page in pages:
        text = page.get("text", None)

        # if text is none we skip it
        if text is None:
            continue

        # if text is empty we skip it
        if text.strip() == "":
            continue

        # if text is too short like headers or page numbers skip it
        if len(text.strip()) < 20:
            continue

        cleaned.append(page)  # keep valid pages

    after = len(cleaned)
    removed = before - after

    stats = {
        "before": before,
        "after":  after,
        "removed_missing": removed
    }

    print(f"  [Missing Values] Before: {before} | Removed: {removed} | After: {after}")
    return cleaned, stats


# ─────────────────────────────────────────────
# STEP 2 — Remove Duplicates
# ─────────────────────────────────────────────

def remove_duplicates(pages):
    """
    Removes pages that have exactly same text.
    """
    before = len(pages)
    seen_texts = set() # to store unique texts
    unique_pages = []

    for page in pages:
        # Normalize text (lowercase + remove extra spaces)
        normalized = " ".join(page["text"].lower().split())

        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_pages.append(page)
        # else: duplicate, skip

    after = len(unique_pages)
    removed = before - after
    stats = {
        "before":            before,
        "after":             after,
        "removed_duplicates": removed
    }

    print(f"  [Duplicates]     Before: {before} | Removed: {removed} | After: {after}")
    return unique_pages, stats


# ─────────────────────────────────────────────
# STEP 3 — Remove Outliers
# ─────────────────────────────────────────────

def remove_outliers(pages, min_words=10, max_words=1000):
    """
    Removes pages based on word count.
    """
    before = len(pages)
    filtered = []
    too_short = 0
    too_long  = 0

    for page in pages:
        word_count = len(page["text"].split())
        # if page has too few words (skip)
        if word_count < min_words:
            too_short += 1
            continue
        # if page has many words (skip)
        if word_count > max_words:
            too_long += 1
            continue

        # save word count inside page
        page["word_count"] = word_count
        filtered.append(page)

    after = len(filtered)

    stats = {
        "before":     before,
        "after":      after,
        "too_short":  too_short,
        "too_long":   too_long,
        "total_removed_outliers": too_short + too_long
    }

    print(f"  [Outliers]       Before: {before} | Too short: {too_short} | Too long: {too_long} | After: {after}")
    return filtered, stats
# ─────────────────────────────────────────────
# STEP 4 — Clean text noise
# ─────────────────────────────────────────────

def clean_text_noise(pages):
    """
    Cleans unwanted patterns from text.
    """
    cleaned_pages = []

    for page in pages:
        text = page["text"]

        # Remove TOC lines like "9. security of person ............. 4"
        text = re.sub(r'[^\n]*\.{4,}[^\n]*', '', text)

        # Clean up blank lines left behind by TOC removal
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove page numbers like 47
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove long lines like -------
        text = re.sub(r'[-_]{3,}', '', text)

        # Remove spaces from start/end of each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        # Final cleanup
        text = text.strip()

        page["text"] = text
        cleaned_pages.append(page)

    print(f"  [Noise Removal]  Cleaned text in {len(cleaned_pages)} pages")
    return cleaned_pages


# ─────────────────────────────────────────────
# Main function that will run all the steps
# ─────────────────────────────────────────────

def preprocess_document(doc_name):
    """
    Runs full cleaning pipeline on one document.
    """
    input_path = os.path.join(INPUT_DIR, f"{doc_name}_raw.json")

    print(f"\n{'='*55}")
    print(f"  🧹 PREPROCESSING: {doc_name.upper()}")
    print(f"{'='*55}")

    # Load raw extracted pages
    if not os.path.exists(input_path):
        print(f"  ❌ File not found: {input_path}")
        print(f"     Run pdf_extractor.py first!")
        return []

    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"  Loaded {len(pages)} raw pages\n")

    all_stats = {"doc_name": doc_name}

    # Step 1 — Missing Values
    pages, stats1 = handle_missing_values(pages)
    all_stats["missing_values"] = stats1

    # Step 2 — Duplicates
    pages, stats2 = remove_duplicates(pages)
    all_stats["duplicates"] = stats2

    # Step 3 — Outliers
    pages, stats3 = remove_outliers(pages)
    all_stats["outliers"] = stats3

    # Step 4 — Noise Removal
    pages = clean_text_noise(pages)

    print(f"\n  ✅ Final clean pages: {len(pages)}")

    return pages, all_stats


# ─────────────────────────────────────────────
# Run preprocessing for ALL documents
# ─────────────────────────────────────────────
def preprocess_all():
    """
    Runs preprocessing on all 3 legal documents.
    Saves cleaned output and preprocessing report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    documents = ["constitution", "ppc", "crpc"]
    all_cleaned = {}
    all_reports = []

    for doc_name in documents:
        result = preprocess_document(doc_name)

        if not result:
            continue

        pages, stats = result
        all_cleaned[doc_name] = pages
        all_reports.append(stats)

        # Save cleaned document
        output_path = os.path.join(OUTPUT_DIR, f"{doc_name}_clean.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

        print(f"  💾 Saved to: {output_path}")

    # Save preprocessing report
    report_path = os.path.join(OUTPUT_DIR, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    # Print final summary
    print(f"\n{'='*55}")
    print(f"  📊 PREPROCESSING SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Document':<20} {'Raw Pages':<15} {'Clean Pages':<15} {'Removed'}")
    print(f"  {'-'*52}")

    for report in all_reports:
        doc    = report["doc_name"]
        raw    = report["missing_values"]["before"]
        clean  = report["outliers"]["after"]
        removed = raw - clean
        print(f"  {doc:<20} {raw:<15} {clean:<15} {removed}")

    print(f"\n  📁 All cleaned files saved in: {OUTPUT_DIR}/")
    print(f"  📋 Report saved to: {report_path}")
    print(f"{'='*55}")

    return all_cleaned


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    all_cleaned = preprocess_all()
    print(f"\n✅ Preprocessing complete!")