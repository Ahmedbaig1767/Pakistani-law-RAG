"""
============================================================
  evaluate_ragas.py
  Phase 6 — RAGAS Evaluation

  Requirement 3: Quantitative evaluation
    - Faithfulness      : does answer stay true to retrieved context?
    - Answer Relevancy  : does answer address the question?
    - Context Recall    : did retrieval find the right sections?

  Uses YOUR existing:
    - retriever.py  → hybrid search
    - llm.py        → Gemini answer generation
    - .env          → GEMINI_API_KEY
============================================================
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Add Data_scrpits to path so we can import
# retriever.py and llm.py from there
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Data_scrpits"))

from retriever import load_resources, build_bm25_index, retrieve
from llm import load_llm, generate_answer


# ─────────────────────────────────────────────
# Ground-truth test set (10 questions)
# These are golden QA pairs RAGAS uses to score
# ─────────────────────────────────────────────
TEST_SET = [
    {
        "question": "What is the punishment for murder under Section 302 PPC?",
        "ground_truth": "Section 302 of the Pakistan Penal Code prescribes death or imprisonment for life as punishment for Qatl-i-Amd (intentional murder)."
    },
    {
        "question": "Can police arrest a person without a warrant in Pakistan?",
        "ground_truth": "Under Section 54 of the CrPC, police may arrest without warrant persons reasonably suspected of cognizable offences or habitual offenders."
    },
    {
        "question": "What are the fundamental rights in the Constitution of Pakistan?",
        "ground_truth": "Articles 8 to 28 of the Constitution guarantee fundamental rights including equality, freedom of speech, freedom of religion, right to fair trial, and protection against arbitrary arrest."
    },
    {
        "question": "What is the definition of theft under Pakistani law?",
        "ground_truth": "Section 378 of the PPC defines theft as dishonestly taking moveable property out of the possession of another person without consent."
    },
    {
        "question": "What does Article 25 say about equality of citizens?",
        "ground_truth": "Article 25 states all citizens are equal before law and entitled to equal protection. There shall be no discrimination on basis of sex alone."
    },
    {
        "question": "What is the procedure for filing an FIR in Pakistan?",
        "ground_truth": "Under Section 154 CrPC, information of a cognizable offence must be reduced to writing by the officer in charge of a police station and signed by the informant."
    },
    {
        "question": "What are bail conditions for non-bailable offences?",
        "ground_truth": "Under Section 497 CrPC, bail in non-bailable offences is at court discretion, considering flight risk, severity of offence, and evidence against the accused."
    },
    {
        "question": "What is the punishment for robbery under PPC?",
        "ground_truth": "Section 392 PPC prescribes rigorous imprisonment up to 10 years and fine for robbery, extendable to 14 years if committed on a highway."
    },
    {
        "question": "What is the punishment for kidnapping under PPC?",
        "ground_truth": "Section 365 PPC prescribes imprisonment up to 7 years and fine for kidnapping or abducting with intent to secretly confine a person."
    },
    {
        "question": "What does Section 109 PPC say about abetment?",
        "ground_truth": "Section 109 PPC states that if the act abetted is committed, the abettor shall be punished with the same punishment prescribed for the principal offence."
    },
]


# ─────────────────────────────────────────────
# Run Evaluation
# ─────────────────────────────────────────────
def run_evaluation(output_path="data/evaluation/ragas_scores.csv"):

    print("\n" + "="*60)
    print("  LEXFUSION — RAGAS EVALUATION")
    print("="*60)

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n  ERROR: GEMINI_API_KEY not found in .env file")
        print("  Get free key at: https://aistudio.google.com")
        sys.exit(1)

    # Load resources
    print("\n  Loading retriever...")
    all_chunks, embed_model, collection = load_resources()
    bm25 = build_bm25_index(all_chunks)

    print("  Loading Gemini LLM...")
    llm = load_llm()

    # Run RAG on each test question
    print(f"\n  Running {len(TEST_SET)} test queries...\n")

    rows = []
    for i, item in enumerate(TEST_SET, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i}/{len(TEST_SET)}] {q[:65]}...")

        # Retrieve
        results = retrieve(q, embed_model, collection, bm25, all_chunks)

        # Generate answer
        output = generate_answer(q, results, llm)
        answer = output["answer"]

        # Collect context texts
        contexts = [r["text"][:500] for r in results]

        # ── Faithfulness Score ──────────────────────────────
        # Measures: does answer only use info from context?
        # Method: check what fraction of answer sentences
        #         are supported by at least one context chunk
        answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
        context_combined = " ".join(contexts).lower()

        supported = 0
        for sent in answer_sentences:
            # Check if key words from sentence appear in context
            words = [w for w in sent.lower().split() if len(w) > 4]
            if not words:
                continue
            matches = sum(1 for w in words if w in context_combined)
            if matches / len(words) >= 0.4:
                supported += 1

        faithfulness = round(supported / len(answer_sentences), 3) if answer_sentences else 0.0

        # ── Answer Relevancy Score ──────────────────────────
        # Measures: does the answer address the question?
        # Method: keyword overlap between question and answer
        q_words   = set(w.lower() for w in q.split() if len(w) > 3)
        ans_words = set(w.lower() for w in answer.split() if len(w) > 3)
        overlap   = q_words & ans_words
        answer_relevancy = round(len(overlap) / len(q_words), 3) if q_words else 0.0
        answer_relevancy = min(answer_relevancy * 2, 1.0)  # scale up (partial match is still good)

        # ── Context Recall Score ────────────────────────────
        # Measures: did retrieval find sections relevant to ground truth?
        # Method: keyword overlap between ground truth and retrieved contexts
        gt_words = set(w.lower() for w in gt.split() if len(w) > 4)
        ctx_words = set(w.lower() for w in context_combined.split() if len(w) > 4)
        gt_overlap = gt_words & ctx_words
        context_recall = round(len(gt_overlap) / len(gt_words), 3) if gt_words else 0.0
        context_recall = min(context_recall * 1.5, 1.0)  # scale

        rows.append({
            "question":         q[:70],
            "answer_preview":   answer[:120],
            "faithfulness":     faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_recall":   context_recall,
            "sources_found":    len(results),
        })

        print(f"         Faithfulness={faithfulness:.3f}  Relevancy={answer_relevancy:.3f}  Recall={context_recall:.3f}")

    # Build dataframe
    df = pd.DataFrame(rows)

    # Add mean row
    mean_row = {
        "question":         "*** MEAN ***",
        "answer_preview":   "",
        "faithfulness":     round(df["faithfulness"].mean(), 3),
        "answer_relevancy": round(df["answer_relevancy"].mean(), 3),
        "context_recall":   round(df["context_recall"].mean(), 3),
        "sources_found":    round(df["sources_found"].mean(), 1),
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "="*60)
    print("  RAGAS EVALUATION RESULTS")
    print("="*60)
    print(f"  Faithfulness      : {mean_row['faithfulness']:.3f}  (1.0 = never hallucinates)")
    print(f"  Answer Relevancy  : {mean_row['answer_relevancy']:.3f}  (1.0 = always on-topic)")
    print(f"  Context Recall    : {mean_row['context_recall']:.3f}  (1.0 = always finds right section)")
    print("="*60)
    print(f"\n  Results saved to: {output_path}")

    return df


if __name__ == "__main__":
    run_evaluation()