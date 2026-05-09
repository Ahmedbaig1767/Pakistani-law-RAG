"""
============================================================
  evaluate_ragas.py
  Phase 7 — RAGAS-style Evaluation

  Requirement 3: Quantitative evaluation with scores for:
    - Faithfulness      : does answer stay true to context?
    - Answer Relevancy  : does answer address the question?
    - Context Recall    : did retrieval find the right sections?
    - Context Precision : were retrieved chunks actually useful?

  Test Set: 30 questions (10 per document)
    - Constitution : Articles covering rights, structure, elections
    - PPC          : Sections covering crimes and punishments
    - CrPC         : Sections covering procedure and courts

  Uses YOUR existing:
    - retriever.py  → hybrid search
    - llm.py        → Gemini answer generation
============================================================
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Data_scrpits"))

from retriever import load_resources, build_bm25_index, retrieve
from llm import load_llm, generate_answer


# ─────────────────────────────────────────────
# Test Set — 30 Questions (10 per document)
# ─────────────────────────────────────────────

TEST_SET = [

    # ══════════════════════════════════════════
    # CONSTITUTION (10 questions)
    # ══════════════════════════════════════════

    {
        "question":     "What is the state religion of Pakistan?",
        "ground_truth": "According to Article 2 of the Constitution, Islam is the state religion of Pakistan.",
        "doc":          "constitution",
        "difficulty":   "easy"
    },
    {
        "question":     "What are the fundamental rights guaranteed by the Constitution of Pakistan?",
        "ground_truth": "Articles 8 to 28 of the Constitution guarantee fundamental rights including equality before law, freedom of speech, freedom of religion, right to fair trial, protection against arbitrary arrest, freedom of movement, and right to education.",
        "doc":          "constitution",
        "difficulty":   "medium"
    },
    {
        "question":     "What does Article 25 say about equality of citizens?",
        "ground_truth": "Article 25 states all citizens are equal before law and entitled to equal protection of law. There shall be no discrimination on the basis of sex alone.",
        "doc":          "constitution",
        "difficulty":   "easy"
    },
    {
        "question":     "How is the Prime Minister of Pakistan elected?",
        "ground_truth": "Under Article 91, the Prime Minister is elected by the votes of the majority of the total membership of the National Assembly from among its Muslim members.",
        "doc":          "constitution",
        "difficulty":   "medium"
    },
    {
        "question":     "What is high treason under the Constitution of Pakistan?",
        "ground_truth": "Article 6 defines high treason as abrogating, subverting, or suspending the Constitution by use of force or show of force or by any other unconstitutional means.",
        "doc":          "constitution",
        "difficulty":   "medium"
    },
    {
        "question":     "What safeguards does the Constitution provide against arrest and detention?",
        "ground_truth": "Article 10 provides that no person shall be arrested without being informed of the grounds, shall be produced before a magistrate within 24 hours, and shall have the right to consult a legal practitioner.",
        "doc":          "constitution",
        "difficulty":   "medium"
    },
    {
        "question":     "What does Article 19 say about freedom of speech?",
        "ground_truth": "Article 19 guarantees every citizen the right to freedom of speech and expression, subject to reasonable restrictions in the interest of the glory of Islam, security of Pakistan, public order, or decency.",
        "doc":          "constitution",
        "difficulty":   "easy"
    },
    {
        "question":     "What is the procedure for election of the President of Pakistan?",
        "ground_truth": "Under Article 41, the President is elected by members of an Electoral College consisting of members of both Houses of Parliament and members of the Provincial Assemblies.",
        "doc":          "constitution",
        "difficulty":   "hard"
    },
    {
        "question":     "Can fundamental rights be suspended during an emergency in Pakistan?",
        "ground_truth": "Article 233 provides that during a Proclamation of Emergency, the President may suspend the enforcement of fundamental rights, except those related to personal liberty under Article 10.",
        "doc":          "constitution",
        "difficulty":   "hard"
    },
    {
        "question":     "What does Article 14 say about the dignity of man?",
        "ground_truth": "Article 14 declares that the dignity of man and the privacy of home shall be inviolable, and no person shall be subjected to torture for the purpose of extracting evidence.",
        "doc":          "constitution",
        "difficulty":   "easy"
    },


    # ══════════════════════════════════════════
    # PPC — Pakistan Penal Code (10 questions)
    # ══════════════════════════════════════════

    {
        "question":     "What is the punishment for murder under Section 302 PPC?",
        "ground_truth": "Section 302 PPC prescribes death or imprisonment for life as punishment for Qatl-i-Amd (intentional murder), along with liability to pay Diyat.",
        "doc":          "ppc",
        "difficulty":   "easy"
    },
    {
        "question":     "What is the definition of theft under the Pakistan Penal Code?",
        "ground_truth": "Section 378 PPC defines theft as dishonestly taking moveable property out of the possession of any person without that person's consent.",
        "doc":          "ppc",
        "difficulty":   "easy"
    },
    {
        "question":     "What is the punishment for robbery under PPC?",
        "ground_truth": "Section 392 PPC prescribes rigorous imprisonment up to 10 years and fine for robbery, which may extend to 14 years if committed on a highway between sunset and sunrise.",
        "doc":          "ppc",
        "difficulty":   "medium"
    },
    {
        "question":     "What does Section 109 PPC say about punishment of abetment?",
        "ground_truth": "Section 109 PPC states that if the act abetted is committed in consequence of the abetment, the abettor shall be punished with the same punishment prescribed for the principal offence.",
        "doc":          "ppc",
        "difficulty":   "medium"
    },
    {
        "question":     "What is the punishment for kidnapping under PPC?",
        "ground_truth": "Section 365 PPC prescribes imprisonment up to 7 years and fine for kidnapping or abducting any person with intent to cause that person to be secretly confined.",
        "doc":          "ppc",
        "difficulty":   "medium"
    },
    {
        "question":     "What is waging war against Pakistan under PPC?",
        "ground_truth": "Section 121 PPC states that whoever wages war against Pakistan or attempts to wage such war or abets waging of such war shall be punished with death or imprisonment for life.",
        "doc":          "ppc",
        "difficulty":   "medium"
    },
    {
        "question":     "What is the legal defence of unsound mind under PPC?",
        "ground_truth": "Section 84 PPC provides that nothing is an offence if done by a person who, at the time of the act, by reason of unsoundness of mind, was incapable of knowing the nature of the act or that it was wrong.",
        "doc":          "ppc",
        "difficulty":   "hard"
    },
    {
        "question":     "What does Section 34 PPC say about common intention?",
        "ground_truth": "Section 34 PPC states that when a criminal act is done by several persons in furtherance of the common intention of all, each such person is liable for that act in the same manner as if it were done by him alone.",
        "doc":          "ppc",
        "difficulty":   "medium"
    },
    {
        "question":     "What is the right of private defence of body under Section 100 PPC?",
        "ground_truth": "Section 100 PPC states the right of private defence of the body extends to causing death if the offence reasonably apprehended is assault which may cause death, grievous hurt, rape, abduction, or wrongful confinement.",
        "doc":          "ppc",
        "difficulty":   "hard"
    },
    {
        "question":     "What is the punishment for criminal intimidation under Section 506 PPC?",
        "ground_truth": "Section 506 PPC prescribes imprisonment up to 2 years or fine or both for criminal intimidation, which may extend to 7 years if the threat is to cause death or grievous hurt.",
        "doc":          "ppc",
        "difficulty":   "easy"
    },


    # ══════════════════════════════════════════
    # CrPC — Criminal Procedure Code (10 questions)
    # ══════════════════════════════════════════

    {
        "question":     "Can police arrest a person without a warrant in Pakistan?",
        "ground_truth": "Under Section 54 CrPC, a police officer may arrest without a warrant any person reasonably suspected of being concerned in a cognizable offence or against whom credible information has been received.",
        "doc":          "crpc",
        "difficulty":   "easy"
    },
    {
        "question":     "What is the procedure for filing an FIR in Pakistan?",
        "ground_truth": "Under Section 154 CrPC, information of a cognizable offence must be reduced to writing by the officer in charge of the police station, read over to the informant, and signed by them, then recorded in a register.",
        "doc":          "crpc",
        "difficulty":   "easy"
    },
    {
        "question":     "What are the bail conditions for non-bailable offences in Pakistan?",
        "ground_truth": "Under Section 497 CrPC, bail in non-bailable offences is at the discretion of the court, considering factors like flight risk, severity of the offence, and the evidence against the accused.",
        "doc":          "crpc",
        "difficulty":   "medium"
    },
    {
        "question":     "Within what time must an arrested person be produced before a magistrate?",
        "ground_truth": "Section 60 CrPC requires that a person arrested without warrant must be brought before a magistrate or officer in charge of a police station without unnecessary delay and within 24 hours.",
        "doc":          "crpc",
        "difficulty":   "easy"
    },
    {
        "question":     "What powers does a magistrate have to issue a search warrant?",
        "ground_truth": "Under Section 98 CrPC, a magistrate may issue a search warrant for any place suspected to contain stolen property, forged documents, counterfeit currency, or objectionable articles.",
        "doc":          "crpc",
        "difficulty":   "medium"
    },
    {
        "question":     "What is the procedure for investigation of a cognizable offence by police?",
        "ground_truth": "Section 156 CrPC empowers a police officer in charge of a police station to investigate any cognizable offence without the order of a magistrate, including proceeding to the spot and investigating the facts.",
        "doc":          "crpc",
        "difficulty":   "medium"
    },
    {
        "question":     "What is the power of magistrate to impose security for keeping the peace?",
        "ground_truth": "Section 106 CrPC empowers a court convicting a person of certain offences to order the offender to execute a bond with sureties for keeping the peace for a period not exceeding 3 years.",
        "doc":          "crpc",
        "difficulty":   "hard"
    },
    {
        "question":     "What does Section 144 CrPC say about urgent cases of nuisance?",
        "ground_truth": "Section 144 CrPC empowers a magistrate to issue an order absolute at once in urgent cases of nuisance or apprehended danger, directing any person to abstain from a certain act, applicable to the public generally.",
        "doc":          "crpc",
        "difficulty":   "hard"
    },
    {
        "question":     "Can statements made to police be used as evidence in court?",
        "ground_truth": "Section 162 CrPC provides that statements made to a police officer during investigation shall not be signed and shall not be used as evidence, except to contradict a witness in court.",
        "doc":          "crpc",
        "difficulty":   "medium"
    },
    {
        "question":     "What is the power of a commissioned military officer to disperse an unlawful assembly?",
        "ground_truth": "Section 131 CrPC empowers a commissioned military officer to disperse an unlawful assembly if it cannot otherwise be dispersed and if immediate action is required to prevent violence or harm to the public.",
        "doc":          "crpc",
        "difficulty":   "hard"
    },
]


# ─────────────────────────────────────────────
# Scoring Functions
# ─────────────────────────────────────────────

def score_faithfulness(answer, contexts):
    """
    Faithfulness: does the answer stay within the retrieved context?
    Method: check what fraction of answer sentences are
            supported by key words from the context.
    Score: 0.0 (hallucinates) → 1.0 (fully grounded)
    """
    answer_sentences = [
        s.strip() for s in answer.split(".")
        if len(s.strip()) > 10
    ]
    if not answer_sentences:
        return 0.0

    context_combined = " ".join(contexts).lower()
    supported = 0

    for sent in answer_sentences:
        words   = [w for w in sent.lower().split() if len(w) > 4]
        if not words:
            continue
        matches = sum(1 for w in words if w in context_combined)
        if matches / len(words) >= 0.35:
            supported += 1

    return round(supported / len(answer_sentences), 3)


def score_answer_relevancy(question, answer):
    """
    Answer Relevancy: does the answer address the question?
    Method: keyword overlap between question and answer.
    Score: 0.0 (off-topic) → 1.0 (directly answers question)
    """
    q_words   = set(w.lower() for w in question.split() if len(w) > 3)
    ans_words = set(w.lower() for w in answer.split()   if len(w) > 3)

    if not q_words:
        return 0.0

    overlap  = q_words & ans_words
    raw      = len(overlap) / len(q_words)
    scaled   = min(raw * 2.5, 1.0)   # scale: partial overlap is still relevant
    return round(scaled, 3)


def score_context_recall(ground_truth, contexts):
    """
    Context Recall: did retrieval find sections relevant to ground truth?
    Method: keyword overlap between ground truth and retrieved context.
    Score: 0.0 (wrong sections) → 1.0 (perfect retrieval)
    """
    gt_words  = set(w.lower() for w in ground_truth.split() if len(w) > 4)
    ctx_text  = " ".join(contexts).lower()
    ctx_words = set(w.lower() for w in ctx_text.split()     if len(w) > 4)

    if not gt_words:
        return 0.0

    overlap = gt_words & ctx_words
    raw     = len(overlap) / len(gt_words)
    scaled  = min(raw * 1.8, 1.0)
    return round(scaled, 3)


def score_context_precision(question, contexts):
    """
    Context Precision: were the retrieved chunks actually useful?
    Method: check how many retrieved chunks contain question keywords.
    Score: 0.0 (all noise) → 1.0 (all chunks relevant)
    """
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    if not q_words or not contexts:
        return 0.0

    useful = 0
    for ctx in contexts:
        ctx_words = set(w.lower() for w in ctx.split() if len(w) > 3)
        overlap   = q_words & ctx_words
        if len(overlap) / len(q_words) >= 0.2:
            useful += 1

    return round(useful / len(contexts), 3)


# ─────────────────────────────────────────────
# Run Evaluation
# ─────────────────────────────────────────────

def run_evaluation(output_path="data/evaluation/ragas_scores.csv"):

    print("\n" + "="*65)
    print("  LEXFUSION — RAGAS EVALUATION (30 Questions)")
    print("="*65)

    # Load resources
    print("\n  Loading retriever...")
    all_chunks, embed_model, collection = load_resources()
    bm25 = build_bm25_index(all_chunks)

    print("  Loading Gemini LLM...")
    llm = load_llm()

    print(f"\n  Running {len(TEST_SET)} test queries...\n")
    print(f"  {'#':<4} {'Doc':<15} {'Diff':<8} {'Faith':>7} {'Relev':>7} {'Recall':>8} {'Precis':>8}")
    print(f"  {'-'*60}")

    rows = []

    for i, item in enumerate(TEST_SET, 1):
        q    = item["question"]
        gt   = item["ground_truth"]
        doc  = item["doc"]
        diff = item["difficulty"]

        # Retrieve
        results = retrieve(q, embed_model, collection, bm25, all_chunks)

        # Generate answer
        output = generate_answer(q, results, llm)
        answer = output["answer"]

        # Collect context texts (cap at 500 chars each for scoring)
        contexts = [r["text"][:500] for r in results]

        # Score
        faithfulness      = score_faithfulness(answer, contexts)
        answer_relevancy  = score_answer_relevancy(q, answer)
        context_recall    = score_context_recall(gt, contexts)
        context_precision = score_context_precision(q, contexts)

        # Top source
        top_source = results[0]["title"] if results else "None"

        rows.append({
            "question":          q[:70],
            "document":          doc,
            "difficulty":        diff,
            "answer_preview":    answer[:150],
            "top_source":        top_source[:60],
            "sources_found":     len(results),
            "faithfulness":      faithfulness,
            "answer_relevancy":  answer_relevancy,
            "context_recall":    context_recall,
            "context_precision": context_precision,
            "avg_score":         round(
                (faithfulness + answer_relevancy + context_recall + context_precision) / 4, 3
            )
        })

        print(f"  {i:<4} {doc:<15} {diff:<8} {faithfulness:>7.3f} {answer_relevancy:>7.3f} "
              f"{context_recall:>8.3f} {context_precision:>8.3f}")

    # ─────────────────────────────────────────
    # Build DataFrame
    # ─────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Per-document averages
    print(f"\n  {'='*65}")
    print(f"  RESULTS BY DOCUMENT")
    print(f"  {'='*65}")
    print(f"  {'Doc':<20} {'Faith':>8} {'Relev':>8} {'Recall':>8} {'Precis':>8} {'Avg':>8}")
    print(f"  {'-'*60}")

    for doc in ["constitution", "ppc", "crpc"]:
        sub = df[df["document"] == doc]
        print(f"  {doc:<20} "
              f"{sub['faithfulness'].mean():>8.3f} "
              f"{sub['answer_relevancy'].mean():>8.3f} "
              f"{sub['context_recall'].mean():>8.3f} "
              f"{sub['context_precision'].mean():>8.3f} "
              f"{sub['avg_score'].mean():>8.3f}")

    # Per-difficulty averages
    print(f"\n  {'='*65}")
    print(f"  RESULTS BY DIFFICULTY")
    print(f"  {'='*65}")
    print(f"  {'Difficulty':<20} {'Faith':>8} {'Relev':>8} {'Recall':>8} {'Precis':>8} {'Avg':>8}")
    print(f"  {'-'*60}")

    for diff in ["easy", "medium", "hard"]:
        sub = df[df["difficulty"] == diff]
        print(f"  {diff:<20} "
              f"{sub['faithfulness'].mean():>8.3f} "
              f"{sub['answer_relevancy'].mean():>8.3f} "
              f"{sub['context_recall'].mean():>8.3f} "
              f"{sub['context_precision'].mean():>8.3f} "
              f"{sub['avg_score'].mean():>8.3f}")

    # Overall averages
    mean_faith  = round(df["faithfulness"].mean(),      3)
    mean_relev  = round(df["answer_relevancy"].mean(),  3)
    mean_recall = round(df["context_recall"].mean(),    3)
    mean_precis = round(df["context_precision"].mean(), 3)
    mean_avg    = round(df["avg_score"].mean(),         3)

    # Add mean row to CSV
    mean_row = {
        "question":          "*** OVERALL MEAN ***",
        "document":          "all",
        "difficulty":        "all",
        "answer_preview":    "",
        "top_source":        "",
        "sources_found":     round(df["sources_found"].mean(), 1),
        "faithfulness":      mean_faith,
        "answer_relevancy":  mean_relev,
        "context_recall":    mean_recall,
        "context_precision": mean_precis,
        "avg_score":         mean_avg,
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Final summary
    print(f"\n  {'='*65}")
    print(f"  OVERALL RAGAS SCORES")
    print(f"  {'='*65}")
    print(f"  Faithfulness       : {mean_faith:.3f}   (1.0 = never hallucinates)")
    print(f"  Answer Relevancy   : {mean_relev:.3f}   (1.0 = always on-topic)")
    print(f"  Context Recall     : {mean_recall:.3f}   (1.0 = always finds right section)")
    print(f"  Context Precision  : {mean_precis:.3f}   (1.0 = all retrieved chunks useful)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Overall Average    : {mean_avg:.3f}")
    print(f"  {'='*65}")
    print(f"\n  Results saved to: {output_path}\n")

    return df


if __name__ == "__main__":
    run_evaluation()