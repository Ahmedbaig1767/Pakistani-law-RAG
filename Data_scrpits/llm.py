"""
============================================================
  llm.py — Updated to use new google-genai package
  Phase 6 — Answer Generation
============================================================
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load API Key from .env file
# ─────────────────────────────────────────────

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ─────────────────────────────────────────────
# Load LLM
# ─────────────────────────────────────────────

def load_llm():
    """
    Configures and returns the Gemini client.
    Uses new google-genai package instead of
    deprecated google-generativeai.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found!\n"
            "1. Go to https://aistudio.google.com\n"
            "2. Create a free API key\n"
            "3. Add to .env file: GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini client loaded successfully")
    return client


# ─────────────────────────────────────────────
# Build Prompt
# ─────────────────────────────────────────────

def build_prompt(query, retrieved_chunks):
    """
    Builds a clean prompt sending FULL chunk text to Gemini.
    No truncation — Gemini handles large context easily.
    """
    if not retrieved_chunks:
        return None

    top_chunks = retrieved_chunks[:3]

    context_blocks = []
    for i, chunk in enumerate(top_chunks, 1):
        doc      = chunk.get("doc_name", "").upper()
        title    = chunk.get("title", "")
        full_text = chunk["text"]

        context_blocks.append(
            f"[SOURCE {i} — {doc} | {title}]\n{full_text}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a legal assistant specializing in Pakistani law.
You have access to the Constitution of Pakistan, Pakistan Penal Code (PPC),
and Criminal Procedure Code (CrPC).

Answer the question using ONLY the legal context provided below.

Rules:
- Give a complete and detailed answer covering ALL points in the context
- Read the ENTIRE context before answering — do not stop at the first sentence
- Cite ALL relevant Article/Section numbers mentioned
- If multiple clauses exist like (1), (2), (3) — explain all of them
- Use simple and clear English that a non-lawyer can understand
- Structure your answer clearly with proper sentences
- If the answer is not in the context, reply exactly: "Not found in the provided context."

Question: {query}

Legal Context:
{context}

Answer:"""

    return prompt.strip()


# ─────────────────────────────────────────────
# Generate Answer
# ─────────────────────────────────────────────

def generate_answer(query, retrieved_chunks, llm):
    """
    Sends prompt to Gemini and returns structured answer.

    Parameters:
        query            (str)  : User's legal question
        retrieved_chunks (list) : Top chunks from retriever
        llm                     : Gemini client from load_llm()

    Returns:
        {
            "answer":  "Complete legal answer...",
            "sources": [{"doc", "title", "score"}, ...]
        }
    """
    if not retrieved_chunks:
        return {
            "answer":  "No relevant legal context found for your question.",
            "sources": []
        }

    prompt = build_prompt(query, retrieved_chunks)

    try:
        response = llm.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
                top_p=0.8,
            )
        )
        answer_text = response.text.strip()

        if len(answer_text.split()) < 5:
            answer_text = "Could not generate a complete answer. Please try rephrasing your question."

    except Exception as e:
        answer_text = f"Generation error: {str(e)[:150]}"

    sources = [
        {
            "doc":   c.get("doc_name", ""),
            "title": c.get("title", ""),
            "score": c.get("hybrid_score", 0.0)
        }
        for c in retrieved_chunks[:3]
    ]

    return {
        "answer":  answer_text,
        "sources": sources
    }


# ─────────────────────────────────────────────
# Print Answer
# ─────────────────────────────────────────────

def print_answer(query, result):
    """Pretty prints the answer and sources to terminal."""
    print("\n" + "=" * 70)
    print(f"  QUESTION: {query}")
    print("=" * 70)
    print(f"\n  ANSWER:\n")

    for line in result["answer"].split("\n"):
        print(f"  {line}")

    print(f"\n  SOURCES:")
    for i, s in enumerate(result["sources"], 1):
        score = s.get("score", 0)
        print(f"  {i}. {s['doc'].upper()} | {s['title'][:60]} (score={score:.3f})")

    print("=" * 70)


# ─────────────────────────────────────────────
# Run — Interactive Test Mode
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from retriever import load_resources, build_bm25_index, retrieve

        print("\n" + "=" * 60)
        print("  Pakistani Law RAG System")
        print("  Powered by Gemini 2.5 Flash Lite")
        print("=" * 60)

        print("\n  Loading resources...")
        all_chunks, embed_model, collection = load_resources()
        bm25 = build_bm25_index(all_chunks)
        llm  = load_llm()

        print("\n  System ready! Type your legal question below.")
        print("  Type 'quit' to exit.\n")

        while True:
            query = input("  Ask: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\n  Goodbye!")
                break

            print("\n  Searching...")
            results = retrieve(query, embed_model, collection, bm25, all_chunks)

            print("  Generating answer...\n")
            output = generate_answer(query, results, llm)
            print_answer(query, output)

    except ImportError as e:
        print(f"\n  Missing dependency: {e}")
        print("  Run: pip install google-genai python-dotenv sentence-transformers")

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    except Exception as e:
        print(f"\n  Error: {e}")