

import os
import google.generativeai as genai
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
    
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found!\n"
            "1. Go to https://aistudio.google.com\n"
            "2. Create a free API key\n"
            "3. Add to .env file: GEMINI_API_KEY=your_key_here"
        )

    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config={
            "temperature":     0.1,   # Low = more factual, less creative
            "max_output_tokens": 1024, # Enough for detailed legal answers
            "top_p":           0.8,
        }
    )

    print("✅ Gemini 1.5 Flash loaded successfully")
    return model


# ─────────────────────────────────────────────
# Build Prompt
# ─────────────────────────────────────────────

def build_prompt(query, retrieved_chunks):
    """
    Builds a clean prompt sending FULL chunk text to Gemini.

    Unlike Flan-T5, Gemini can handle the full context
    so we do NOT truncate the chunk text anymore.
    """
    if not retrieved_chunks:
        return None

    # Use top 3 chunks for better coverage
    top_chunks = retrieved_chunks[:3]

    context_blocks = []
    for i, chunk in enumerate(top_chunks, 1):
        section  = chunk.get("section_num", "N/A")
        doc      = chunk.get("doc_name", "").upper()
        title    = chunk.get("title", "")

        # Send FULL text — no truncation
        # Gemini 1.5 Flash handles 128k tokens
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
        llm                     : Gemini model from load_llm()

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
        response     = llm.generate_content(prompt)
        answer_text  = response.text.strip()

        # Sanity check — if answer is too short something went wrong
        if len(answer_text.split()) < 5:
            answer_text = "Could not generate a complete answer. Please try rephrasing your question."

    except Exception as e:
        answer_text = f"Generation error: {str(e)[:100]}"

    # Build sources list for UI display
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

    # Print answer with proper indentation
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
        print("  Powered by Gemini 1.5 Flash")
        print("=" * 60)

        # Load everything once
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
        print("  Run: pip install google-generativeai python-dotenv")

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    except Exception as e:
        print(f"\n  Error: {e}")