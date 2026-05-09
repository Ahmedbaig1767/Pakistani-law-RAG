# ============================================================
#   Uses YOUR existing:
#     - retriever.py  → hybrid search
#     - llm.py        → Gemini answer generation
# ============================================================
import html
import os
import re
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────
# Add Data_scrpits to path
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Data_scrpits"))
# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LexFusion — Pakistani Law AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Theme & CSS — custom “app shell” (non-default Streamlit look)
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,400&family=Outfit:wght@300;400;500;600;700&family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,500;1,7..72,400&display=swap');

  :root {
    --ink: #f3eee6;
    --muted: #a39e94;
    --surface: #1c1917;
    --surface2: #26211d;
    --line: rgba(212, 196, 176, 0.14);
    --gold: #d4a574;
    --gold-dim: rgba(212, 165, 116, 0.35);
    --clay: #c45c4a;
    --clay-glow: rgba(196, 92, 74, 0.45);
    --sage: #6b8f71;
    --radius: 20px;
    --font-ui: 'Outfit', system-ui, sans-serif;
    --font-display: 'Fraunces', Georgia, serif;
    --font-body: 'Literata', Georgia, serif;
  }

  /* App canvas — warm charcoal, not Streamlit white/blue */
  .stApp {
    background:
      radial-gradient(ellipse 900px 500px at 15% -10%, rgba(212, 165, 116, 0.12), transparent 55%),
      radial-gradient(ellipse 700px 420px at 95% 5%, rgba(196, 92, 74, 0.08), transparent 50%),
      linear-gradient(165deg, #12100e 0%, #1a1612 40%, #14110f 100%) !important;
    color: var(--ink);
  }

  header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: 1px solid var(--line) !important;
    backdrop-filter: blur(8px);
  }
  div[data-testid="stToolbar"] { display: none !important; }

  .block-container {
    padding-top: 1.1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1320px !important;
  }

  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}

  /* Typography defaults */
  .stMarkdown, .stCaption, label, p, span {
    font-family: var(--font-ui);
  }

  /* Inputs — pill, warm */
  .stTextInput label { color: var(--muted) !important; font-weight: 500 !important; font-size: 0.82rem !important; letter-spacing: 0.04em; text-transform: uppercase !important; }
  div[data-baseweb="input"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }
  div[data-baseweb="input"] input {
    color: var(--ink) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.98rem !important;
  }

  /* Buttons */
  .stButton > button {
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    border-radius: 999px !important;
    border: none !important;
    padding: 0.55rem 1.25rem !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease !important;
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c45c4a 0%, #a8483a 100%) !important;
    color: #fff8f5 !important;
    box-shadow: 0 8px 28px var(--clay-glow) !important;
  }
  .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 36px var(--clay-glow) !important;
  }
  .stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--line) !important;
  }

  /* Select & slider */
  div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--line) !important;
    border-radius: 12px !important;
  }
  .stSlider label { color: var(--muted) !important; }
  div[data-testid="stSlider"] { padding-top: 0.5rem; }
  div[data-baseweb="slider"] [role="slider"] {
    background: var(--gold) !important;
  }

  /* Expanders — card style */
  details {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    overflow: hidden;
  }
  summary {
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    color: var(--gold) !important;
    letter-spacing: 0.02em;
  }
  details summary:hover { color: var(--ink) !important; }

  /* Dataframe chrome */
  div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }

  /* Sidebar — leather / panel */
  div[data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, rgba(28, 25, 23, 0.98) 0%, rgba(18, 16, 14, 0.99) 100%) !important;
    border-right: 1px solid var(--line) !important;
  }
  div[data-testid="stSidebar"] .block-container { padding-top: 1.25rem !important; }
  div[data-testid="stSidebar"] h3 { font-family: var(--font-display) !important; color: var(--ink) !important; font-weight: 600 !important; }
  div[data-testid="stSidebar"] hr { border-color: var(--line) !important; }
  div[data-testid="stSidebar"] .stMarkdown { color: var(--muted); }
  div[data-testid="stSidebar"] .stButton > button {
    background: rgba(212, 165, 116, 0.08) !important;
    color: #ebe4d8 !important;
    border: 1px solid var(--line) !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
  }
  div[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--gold-dim) !important;
    background: rgba(212, 165, 116, 0.14) !important;
  }

  /* Hero */
  .lex-hero {
    position: relative;
    overflow: hidden;
    border-radius: var(--radius);
    padding: 0;
    margin-bottom: 1.6rem;
    background: var(--surface);
    border: 1px solid var(--line);
    box-shadow: 0 28px 80px rgba(0, 0, 0, 0.55);
  }
  .lex-hero-grid {
    display: grid;
    grid-template-columns: 1fr minmax(120px, 28%);
    gap: 0;
    min-height: 168px;
  }
  @media (max-width: 700px) {
    .lex-hero-grid { grid-template-columns: 1fr; }
  }
  .lex-hero-copy {
    padding: 1.65rem 1.75rem 1.5rem;
    position: relative;
    z-index: 1;
  }
  .lex-hero-copy h1 {
    margin: 0;
    font-family: var(--font-display);
    font-size: clamp(1.75rem, 3.2vw, 2.35rem);
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--ink);
  }
  .lex-hero-copy h1 span {
    background: linear-gradient(120deg, var(--gold) 0%, #e8c49a 45%, var(--clay) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .lex-hero-copy p {
    margin: 0.65rem 0 0;
    font-family: var(--font-ui);
    font-size: 0.94rem;
    font-weight: 400;
    color: var(--muted);
    max-width: 46ch;
    line-height: 1.6;
  }
  .lex-hero-art {
    position: relative;
    background:
      radial-gradient(circle at 70% 30%, rgba(212, 165, 116, 0.22), transparent 55%),
      linear-gradient(145deg, #2a221c 0%, #1a1512 100%);
    border-left: 1px solid var(--line);
  }
  .lex-hero-art::before {
    content: "§";
    position: absolute;
    right: 8%;
    bottom: 6%;
    font-family: var(--font-display);
    font-size: clamp(4rem, 12vw, 7rem);
    font-weight: 600;
    color: rgba(212, 165, 116, 0.12);
    line-height: 1;
    pointer-events: none;
  }
  .lex-badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.1rem;
  }
  .lex-pill {
    display: inline-flex;
    align-items: center;
    font-family: var(--font-ui);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ink);
    background: rgba(212, 165, 116, 0.1);
    border: 1px solid var(--gold-dim);
    padding: 0.38rem 0.75rem;
    border-radius: 999px;
  }

  /* Panels */
  .lex-panel-wrap {
    background:
      linear-gradient(180deg, rgba(38, 33, 29, 0.85) 0%, rgba(24, 21, 18, 0.65) 100%);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 1.35rem 1.45rem 1.45rem;
    min-height: 420px;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.35);
    position: relative;
  }
  .lex-panel-wrap::before {
    content: "";
    position: absolute;
    top: 0; left: 24px; right: 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
    opacity: 0.7;
  }
  .lex-panel-head {
    margin-bottom: 1rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--line);
  }
  .lex-panel-title {
    font-family: var(--font-ui);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0;
  }
  .lex-panel-sub {
    font-family: var(--font-ui);
    font-size: 0.8rem;
    font-weight: 400;
    color: var(--muted);
    margin: 0.35rem 0 0;
    line-height: 1.45;
  }
  .lex-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--gold), #8b6914);
    box-shadow: 0 0 0 3px rgba(212, 165, 116, 0.2);
    flex-shrink: 0;
    margin-top: 5px;
  }

  .lex-answer {
    font-family: var(--font-body);
    font-size: 1.03rem;
    line-height: 1.82;
    color: #ebe4d8;
    background: rgba(12, 10, 9, 0.55);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    min-height: 280px;
  }

  .lex-cite-scroll {
    max-height: 520px;
    overflow-y: auto;
    padding-right: 8px;
    margin-right: -4px;
  }
  .lex-cite-scroll::-webkit-scrollbar { width: 7px; }
  .lex-cite-scroll::-webkit-scrollbar-thumb {
    background: rgba(212, 165, 116, 0.28);
    border-radius: 999px;
  }

  .lex-cite-card {
    position: relative;
    background: linear-gradient(165deg, rgba(32, 28, 24, 0.95) 0%, rgba(18, 16, 14, 0.9) 100%);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    border-left: 3px solid var(--clay);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
  }
  .lex-cite-meta {
    font-family: var(--font-ui);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .lex-cite-score {
    font-size: 0.65rem;
    font-weight: 700;
    color: #1a1512;
    background: linear-gradient(135deg, #e8c49a, var(--gold));
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    white-space: nowrap;
  }
  .lex-cite-body {
    font-family: var(--font-body);
    font-size: 0.9rem;
    line-height: 1.68;
    color: #d6cfc4;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .lex-cite-trunc {
    font-family: var(--font-ui);
    font-size: 0.74rem;
    color: var(--muted);
    font-style: italic;
    margin-top: 0.4rem;
  }

  .lex-foot {
    text-align: center;
    color: var(--muted);
    font-family: var(--font-ui);
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
  }

  /* Metrics — editorial numbers */
  div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
  }
  div[data-testid="stMetric"] label { color: var(--muted) !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--gold) !important;
    font-family: var(--font-display) !important;
    font-size: 1.65rem !important;
  }

  /* Info / error — not default blue */
  div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid var(--line) !important;
    background: rgba(38, 33, 29, 0.9) !important;
  }
  div[data-testid="stAlert"] p, div[data-testid="stAlert"] div {
    color: var(--ink) !important;
    font-family: var(--font-ui) !important;
  }

  /* Spinner text */
  .stSpinner > div { color: var(--gold) !important; }

  .stCaption, [data-testid="stCaption"] {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


def _escape_html(s: str) -> str:
    return html.escape(s or "", quote=True)


def _answer_to_html(answer: str) -> str:
    """Preserve line breaks; escape HTML for safety."""
    return _escape_html(answer).replace("\n", "<br>")


def _highlight_query_words(text: str, query: str, max_len: int = 1000) -> str:
    chunk = (text or "")[:max_len]
    highlighted = _escape_html(chunk)
    q = (query or "").lower().split()
    for word in q:
        if len(word) <= 3:
            continue
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            r'<mark style="background:rgba(107,143,113,0.42);color:#e8f0e9;padding:0 3px;border-radius:4px">\g<0></mark>',
            highlighted,
        )
    if len(text or "") > max_len:
        highlighted += (
            f'<div class="lex-cite-trunc">… [{len(text) - max_len:,} more characters]</div>'
        )
    return highlighted


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="lex-hero">
  <div class="lex-hero-grid">
    <div class="lex-hero-copy">
      <h1><span>LexFusion</span></h1>
      <p>Constitution, PPC &amp; CrPC — hybrid retrieval with grounded answers beside the exact statute text.</p>
      <div class="lex-badge-row">
        <span class="lex-pill">Dual panel</span>
        <span class="lex-pill">Hybrid search</span>
        <span class="lex-pill">Gemini</span>
      </div>
    </div>
    <div class="lex-hero-art" aria-hidden="true"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Fraunces,serif;font-size:1.35rem;font-weight:600;margin:0 0 0.2rem;color:#f3eee6">Studio</p>', unsafe_allow_html=True)
    st.caption("Corpus & retrieval")
    doc_filter = st.selectbox(
        "Filter by document",
        ["All Documents", "constitution", "ppc", "crpc"],
    )
    doc_filter = None if doc_filter == "All Documents" else doc_filter

    top_k = st.slider("Sources to retrieve", 3, 10, 5)

    st.markdown("---")
    st.markdown('<p style="font-family:Fraunces,serif;font-size:1.1rem;font-weight:600;margin:0;color:#f3eee6">Try a question</p>', unsafe_allow_html=True)
    examples = [
        "What is the punishment for murder under Section 302?",
        "Can police arrest without a warrant?",
        "What are the fundamental rights in the Constitution?",
        "What is the definition of theft under PPC?",
        "What are bail conditions under CrPC?",
        "What does Article 25 say about equality?",
        "What is the punishment for robbery?",
        "How is an FIR filed in Pakistan?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query"] = ex

    st.markdown("---")
    st.caption("Chroma · embeddings · Gemini")

# ─────────────────────────────────────────────
# Load pipeline once (cached)
# ─────────────────────────────────────────────
@st.cache_resource(
    show_spinner="Loading LexFusion — first load can take ~30 seconds…"
)
def load_pipeline():
    from retriever import build_bm25_index, load_resources
    from llm import load_llm

    all_chunks, embed_model, collection = load_resources()
    bm25 = build_bm25_index(all_chunks)
    llm = load_llm()
    return all_chunks, embed_model, collection, bm25, llm


# ─────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────
query = st.text_input(
    "Your question",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What is the punishment for robbery under Pakistani law?",
    key="query_input",
)
st.caption("Sidebar has one-tap examples · results open in two columns after Search.")

col1, col2 = st.columns([5, 1])
with col1:
    ask = st.button("Search", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state["query"] = ""
        st.rerun()

# ─────────────────────────────────────────────
# Run query
# ─────────────────────────────────────────────
if ask and query.strip():

    if not os.getenv("GEMINI_API_KEY"):
        st.error(
            "GEMINI_API_KEY not found in .env. Get a key at https://aistudio.google.com"
        )
        st.stop()

    with st.spinner("Searching statutes and drafting the answer…"):
        try:
            from llm import generate_answer
            from retriever import retrieve

            all_chunks, embed_model, collection, bm25, llm = load_pipeline()

            results = retrieve(
                query=query.strip(),
                model=embed_model,
                collection=collection,
                bm25=bm25,
                all_chunks=all_chunks,
                top_k=top_k,
                doc_filter=doc_filter,
            )

            output = generate_answer(query.strip(), results, llm)
            answer = output["answer"]

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ─────────────────────────────────────────
    # Side-by-Side Layout (Requirement 4)
    # ─────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
<div class="lex-panel-wrap">
  <div class="lex-panel-head">
    <div style="display:flex;gap:0.65rem;align-items:flex-start;">
      <div class="lex-dot"></div>
      <div>
        <p class="lex-panel-title">AI answer</p>
        <p class="lex-panel-sub">Grounded on retrieved sections; verify critical matters with a qualified lawyer.</p>
      </div>
    </div>
  </div>
  <div class="lex-answer">"""
            + _answer_to_html(answer)
            + """</div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("Hybrid search scores"):
            diag = []
            for r in results:
                diag.append(
                    {
                        "Section": r.get("title", "")[:55],
                        "Doc": r.get("doc_name", "").upper(),
                        "Hybrid Score": r.get("hybrid_score", 0),
                        "Semantic Score": r.get("semantic_score", 0),
                        "Keyword Score": r.get("keyword_score", 0),
                    }
                )
            st.dataframe(
                pd.DataFrame(diag), hide_index=True, use_container_width=True
            )

    with right:
        st.markdown(
            """
<div class="lex-panel-wrap">
  <div class="lex-panel-head">
    <div style="display:flex;gap:0.65rem;align-items:flex-start;">
      <div class="lex-dot" style="background:linear-gradient(135deg,#c45c4a,#8f3d32);box-shadow:0 0 0 3px rgba(196,92,74,0.25)"></div>
      <div>
        <p class="lex-panel-title">Retrieved legal text</p>
        <p class="lex-panel-sub">Exact snippets from your corpus as returned by hybrid search (keywords highlighted).</p>
      </div>
    </div>
  </div>
  <div class="lex-cite-scroll">
""",
            unsafe_allow_html=True,
        )

        qstrip = query.strip()
        cards_html = []
        for i, src in enumerate(results, 1):
            title = src.get("title", f"Source {i}")
            doc = src.get("doc_name", "").upper()
            score = float(src.get("hybrid_score", 0) or 0)
            text = src.get("text", "")
            highlighted = _highlight_query_words(text, qstrip, max_len=1000)

            cards_html.append(
                f"""
<div class="lex-cite-card">
  <div class="lex-cite-meta">
    <span>[{i}] {_escape_html(doc)} · {_escape_html(title[:80])}</span>
    <span class="lex-cite-score">{score:.4f}</span>
  </div>
  <div class="lex-cite-body">{highlighted}</div>
</div>
"""
            )

        st.markdown("".join(cards_html) + "</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RAGAS Scores Panel
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("RAGAS evaluation scores", expanded=False):
    scores_path = Path("data/evaluation/ragas_scores.csv")
    if scores_path.exists():
        df = pd.read_csv(scores_path)
        mean_row = df[df["question"] == "*** MEAN ***"]
        if not mean_row.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Faithfulness",
                    f"{float(mean_row['faithfulness'].iloc[0]):.3f}",
                    help="Does the answer stay true to retrieved law? (1.0 = best)",
                )
            with c2:
                st.metric(
                    "Answer relevancy",
                    f"{float(mean_row['answer_relevancy'].iloc[0]):.3f}",
                    help="Does it address the question? (1.0 = best)",
                )
            with c3:
                st.metric(
                    "Context recall",
                    f"{float(mean_row['context_recall'].iloc[0]):.3f}",
                    help="Did retrieval surface the right section? (1.0 = best)",
                )

        st.markdown("**Per-question breakdown**")
        st.dataframe(
            df[df["question"] != "*** MEAN ***"][
                ["question", "faithfulness", "answer_relevancy", "context_recall"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "Run `python evaluate_ragas.py` to generate scores, then refresh this page."
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    """
<hr style="margin-top:2.25rem;border:none;border-top:1px solid rgba(212,196,176,0.14)">
<p class="lex-foot">LexFusion · FAST University Lahore · Constitution · PPC · CrPC</p>
""",
    unsafe_allow_html=True,
)
