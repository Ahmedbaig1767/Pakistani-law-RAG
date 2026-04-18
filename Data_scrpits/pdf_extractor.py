

# import necessary libraries
import pdfplumber  #used to read and extract text from pdf
import os          #for file operations
import json        #to save in json form


# ─────────────────────────────────────────────
# Pdf File paths (input files)
# ─────────────────────────────────────────────
PDF_FILES = {
    "constitution": "data/raw/PakLaw(NAP).pdf",
    "ppc":          "data/raw/PakistanPenalCode(UNODC).pdf",
    "crpc":         "data/raw/CodeofCriminalProcedure.pdf"
}
#folder where extracted data is saved
OUTPUT_DIR = "data/extracted"


# ─────────────────────────────────────────────
# Function to extract data from 1 pdf
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path, doc_name):
    """
    This function reads a pdf file and extracts text page by page
    Input: 
    pdf_path (location of pdf file)
    doc_name (name of doc like ppc)
    Output:
    List of dictionaries (each dict = one page)
    """
    pages_data = [] #list to store extracted pages

    print(f"\n📄 Extracting: {doc_name} from {pdf_path}")
    # check if file exists
    if not os.path.exists(pdf_path):
        print(f"  ❌ File not found: {pdf_path}")
        return pages_data
    #open pdf using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"  Total pages: {total_pages}")
        #loop through each page
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            # Only keep pages that have text
            if text and text.strip():
                pages_data.append({
                    "doc_name":    doc_name,
                    "page_number": i + 1,
                    "text":        text.strip()
                })

        print(f"  ✅ Extracted {len(pages_data)} pages with text")

    return pages_data

# ─────────────────────────────────────────────
# Function: Extract ALL PDFs
# ─────────────────────────────────────────────
def extract_all_documents():
    """
    This function runs extraction for all PDFs
    and saves results into JSON files.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_documents = {} # store all extracted data
    extraction_summary = [] # store stats (pages, words, etc.)

    print("=" * 55)
    print("  📂 STARTING PDF EXTRACTION")
    print("=" * 55)

    # Loop through each document
    for doc_name, pdf_path in PDF_FILES.items():
        pages = extract_text_from_pdf(pdf_path, doc_name)
        all_documents[doc_name] = pages

        # Save extracted pages into JSON file
        output_path = os.path.join(OUTPUT_DIR, f"{doc_name}_raw.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

        # Collect summary stats
        total_chars = sum(len(p["text"]) for p in pages)
        total_words = sum(len(p["text"].split()) for p in pages)
        extraction_summary.append({
            "document":    doc_name,
            "pages":       len(pages),
            "total_words": total_words,
            "total_chars": total_chars,
            "saved_to":    output_path
        })

        print(f"  💾 Saved to: {output_path}")

    # Save combined summary
    summary_path = os.path.join(OUTPUT_DIR, "extraction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(extraction_summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 55)
    print("  📊 EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"  {'Document':<20} {'Pages':<10} {'Words':<15} {'Chars'}")
    print(f"  {'-'*50}")
    for s in extraction_summary:
        print(f"  {s['document']:<20} {s['pages']:<10} {s['total_words']:<15} {s['total_chars']}")

    total_pages = sum(s['pages'] for s in extraction_summary)
    total_words = sum(s['total_words'] for s in extraction_summary)
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<20} {total_pages:<10} {total_words}")
    print("=" * 55)

    return all_documents


# ─────────────────────────────────────────────
# Main execution starts here
# ─────────────────────────────────────────────

if __name__ == "__main__":
    documents = extract_all_documents()
    print(f"\n✅ Extraction complete. Files saved in: {OUTPUT_DIR}/")