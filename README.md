# 🏛️ Lex Fusion — Pakistani Law RAG System

> An intelligent Retrieval-Augmented Generation (RAG) system that answers legal questions based on Pakistani law documents with evidence-backed responses.

---

## 📌 Project Overview

Access to legal information in Pakistan is severely limited. Citizens, students, and junior lawyers struggle to navigate complex legal documents like the Pakistan Penal Code, Constitution, and Criminal Procedure Code. Traditional search engines return irrelevant results, and hiring a lawyer for basic legal queries is expensive and inaccessible for the average Pakistani.

**Lex Fusion** solves this by allowing users to ask natural language questions about Pakistani law and receive accurate, grounded answers — with the exact legal section shown as evidence.

---

## 🎯 Target Audience

- Pakistani citizens needing basic legal guidance
- Law students at universities
- Junior lawyers needing fast section references
- NGOs and legal aid organizations
- Journalists researching legal aspects of news stories

---

## 📂 Project Structure

```
Pakistani-law-RAG/
│
├── README.md                        ← Project documentation
├── .gitignore                       ← Ignores PDFs, .env, cache
│
├── Data_scrpits/                    ← Phase 2: Data pipeline
│   ├── pdf_extractor.py             ← Extracts text from 3 legal PDFs
│   ├── preprocessor.py              ← Missing values, duplicates, outliers
│   ├── normalizer.py                ← Normalization, encoding, scaling
│   └── eda.py                       ← EDA charts for report
│
├── data/
│   ├── extracted/                   ← Raw JSON from pdf_extractor.py
│   └── processed/                   ← Clean JSON from preprocessor.py
│
└── Report.tex                       ← LaTeX report
```

---

## 📚 Dataset

| Document | Pages | Words | Sections |
|---|---|---|---|
| Constitution of Pakistan | 222 | 70,506 | 288 Articles |
| Pakistan Penal Code (PPC) | 178 | 87,560 | 410 Sections |
| Criminal Procedure Code (CrPC) | 290 | 102,814 | 752 Sections |
| **Total** | **690** | **260,880** | **1,450+** |

> ⚠️ Raw PDF files are not included in this repository as per submission guidelines. Processed JSON files are available in `data/extracted/` and `data/processed/`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| PDF Extraction | `pdfplumber` |
| Data Processing | `pandas`, `numpy`, `re` |
| NLP | `nltk`, `scikit-learn` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Ahmedbaig1767/Pakistani-law-RAG.git
cd Pakistani-law-RAG
```

### 2. Install dependencies
```bash
pip install pdfplumber pandas numpy nltk scikit-learn matplotlib seaborn wordcloud
```

### 3. Add your PDFs
Place your legal PDFs inside `data/raw/`:
```
data/raw/Pak Law (NAP).pdf
data/raw/Pakistan_Penal_Code (UNODC).pdf
data/raw/CodeofCriminalProcedure.pdf
```

### 4. Run the data pipeline
```bash
# Step 1 — Extract text from PDFs
python Data_scrpits/pdf_extractor.py

# Step 2 — Clean and preprocess
python Data_scrpits/preprocessor.py

# Step 3 — Normalize and encode
python Data_scrpits/normalizer.py

# Step 4 — Generate EDA charts
python Data_scrpits/eda.py
```

---

## 📊 Data Pipeline (Phase 2)

### Collection
- Legal PDFs sourced from official Pakistani government websites
- Extracted using `pdfplumber` page by page
- Saved as structured JSON with `doc_name`, `page_number`, `text`

### Cleaning
| Step | Description |
|---|---|
| Missing Values | Removed empty/blank pages and pages with < 20 characters |
| Duplicates | Removed repeated sections caused by PDF formatting |
| Outliers | Filtered pages with < 10 words (too short) or > 1000 words (too long) |

### Preprocessing Results
| Document | Raw Pages | Clean Pages | Removed |
|---|---|---|---|
| Constitution | 222 | 222 | 0 |
| PPC | 178 | 177 | 1 |
| CrPC | 290 | 290 | 0 |

### Feature Strategy
- **Normalization** → Lowercase, whitespace standardization, special character removal
- **Encoding** → Text converted to dense vector embeddings via `sentence-transformers`
- **Scaling** → Chunk sizes standardized to 10–1000 word range

---

## 🤖 Why RAG and not Rule-Based Search?

Traditional keyword search fails when users phrase legal questions in natural language. For example:

> *"Can police arrest me without proof?"*

A keyword search finds nothing. A RAG system understands the **intent** and retrieves `Section 54 of CrPC` — the section on arrest without warrant. The LLM then synthesizes the retrieved sections into a clear, human-readable answer — something rule-based systems fundamentally cannot do.

---



---

## 📄 License

This project is submitted as part of the AI course at FAST University Lahore (Semester 6).
