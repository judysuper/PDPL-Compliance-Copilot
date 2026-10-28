"""Experiment E1 (LLM-only vs naive RAG vs hybrid + rerank) and retrieval metrics.

Retrieval metrics: Recall@k and MRR@k for BM25, dense, RRF fusion and the reranked
list, reported separately for Arabic and English questions (brief: "Recall@5, MRR,
reported separately for Arabic and English"). No LLM is involved, and nothing in
retrieval was tuned on the gold set, so all answerable questions are used.

E1 runs on the evaluation half (the same 30 questions as the day-3 results), so the
three systems are compared on questions none of them was tuned on:
  llm_only   - the LLM answers from its own knowledge, no sources
  naive_rag  - dense retrieval (multilingual e5) top-5, same prompt as the final system,
               no BM25, no reranker, no threshold, no citation repair
  full       - hybrid BM25 + dense, RRF, cross-encoder rerank, threshold, citation gate
               (read from the day-3 evaluation run; not run again)
"""

import random
import time

import pandas as pd

from evaluation.eval_abstention import _parse_articles, _rate, gold_rank, summarize
from generation.citation_checker import extract_articles, normalize_article_number
from generation.generation import LAST_LLM, OUTPUT_FORMAT, call_llm, is_arabic

# ---------------------------------------------------------------- retrieval metrics
METHODS = (("bm25", "bm25"), ("dense", "dense"), ("rrf", "fused"), ("reranked", "final"))


def retrieval_ranks(cache, gold):
    """One row per answerable question: rank of the gold article in each list (None = missing)."""
    rows = []
    for row in gold[gold["answerable"]].itertuples():
        entry = cache[row.qid]
        out = {"qid": row.qid, "language": row.language, "gold_source": row.gold_source,
               "gold_articles": " | ".join(row.gold_articles)}
        for name, key in METHODS:
            out[f"{name}_rank"] = gold_rank(entry.get(key) or [], row)
        rows.append(out)
    return pd.DataFrame(rows)


def _recall(ranks, k):
    return round(float(ranks.map(lambda r: r is not None and not pd.isna(r) and r <= k).mean()), 3)


def _mrr(ranks, k):
    return round(float(ranks.map(lambda r: 1.0 / r if r is not None and not pd.isna(r) and r <= k
                                 else 0.0).mean()), 3)


def retrieval_table(ranks, k=5):
    """Recall@k and MRR@k per method, for all questions and per language."""
    rows = []
    groups = [("all", ranks)] + [(lang, g) for lang, g in ranks.groupby("language")]
    for name, _ in METHODS:
        for label, g in groups:
            col = g[f"{name}_rank"]
            rows.append({"method": name, "language": label, "n": len(g),
                         f"recall@{k}": _recall(col, k), f"mrr@{k}": _mrr(col, k),
                         "recall@10": _recall(col, 10) if name != "reranked" else None})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- LLM-only
LLM_ONLY_RULES = """You are a Saudi Personal Data Protection Law (PDPL) assistant.

Answer the user's question from your own knowledge of the PDPL, its Implementing Regulation and the Regulation on Personal Data Transfer outside the Kingdom. No legal sources are provided.

Rules:
1. Cite the articles you rely on, written like this: (Article 5), and name the document.
2. If you do not know, or the question asks for facts that legal texts cannot provide (statistics, a specific company or person), begin your reply with the exact tag NOT_COVERED: followed by a short explanation in the user's language.
3. If the question is in Arabic, answer in Arabic. If it is in English, answer in English.
4. This is general information, not legal advice."""


class LLMOnly:
    """Pipeline-like object (same .ask interface as RAGPipeline) that sends no sources."""

    def __init__(self, provider=None):
        self.provider = provider

    def ask(self, question, provider=None, retrieved=None, **kwargs):
        from generation.abstention import NOT_COVERED_RE, is_not_covered

        provider = provider or self.provider
        t = time.perf_counter()
        user = f"User question:\n{question}\n\n{OUTPUT_FORMAT}"
        if provider == "mock":
            answer = "[MOCK] (Article 5)"
        else:
            answer = call_llm(user, provider, system=LLM_ONLY_RULES)
        abstained = is_not_covered(answer)
        lang = "ar" if is_arabic(question) else "en"
        return {
            "question": question, "language": lang, "answer": answer, "raw_answer": answer,
            "abstained": abstained, "abstention_reason": "llm_not_covered" if abstained else None,
            "citation_check": None, "first_attempt_valid": None, "n_attempts": 1,
            "provider": provider, "llm_model": LAST_LLM.get("model"), "prompt_style": "llm_only",
            "threshold": None, "max_rerank_score": None, "final_sources": [],
            "timings": {"total_s": time.perf_counter() - t},
            "detail": NOT_COVERED_RE.sub("", answer.strip(), count=1) if abstained else None,
        }


# ------------------------------------------------------------------- naive RAG
def naive_retrieval(dense, gold_df, top_k=5):
    """Dense-only top-k per question, in the cache format run_eval expects."""
    from retrieval.hybrid_retrieval import detect_language

    cache = {}
    for row in gold_df.itertuples():
        t = time.perf_counter()
        lang = detect_language(row.question)
        docs = dense.search(row.question, top_k=top_k, language=lang)
        cache[row.qid] = {"language": lang, "bm25": [], "dense": docs, "fused": docs, "final": docs,
                          "max_rerank_score": None, "timings": {"retrieval_s": time.perf_counter() - t}}
    return cache


NAIVE_ASK_KWARGS = {"min_rerank_score": None, "enforce_citations": False, "max_retries": 0}


# ---------------------------------------------------------------- E1 summary
def _cited_numbers(answer):
    return set(extract_articles(answer or ""))


def e1_row(system, rows, manual=None):
    """One E1 table row from run_eval rows (and optional manual correctness)."""
    s = summarize(rows)
    ans = rows[rows["expected"] == "answer"]
    answered = ans[~ans["abstained"].astype(bool)]
    if system == "llm_only":
        # no sources: lenient check that the gold article NUMBER is cited (any document)
        gold_hit = answered.apply(
            lambda r: bool(set(_parse_articles(r["gold_articles"])) & _cited_numbers(r["raw_answer"])),
            axis=1)
        cites_gold = _rate(gold_hit.astype(float)) if len(answered) else None
        citation_valid = None
        gold_in_sources = None
    else:
        cites_gold = s["cites_gold_article_rate"]
        citation_valid = s["first_attempt_citation_valid_rate"]
        gold_in_sources = s["gold_in_final_sources_rate"]
    row = {
        "system": system,
        "n_questions": s["n_questions"],
        "abstention_accuracy": s["abstention_accuracy"],
        "false_refusal_rate": s["false_refusal_rate"],
        "citations_valid_first_attempt": citation_valid,
        "cites_gold_article": cites_gold,
        "gold_in_sources": gold_in_sources,
        "n_errors": s["n_errors"],
    }
    if manual is not None:
        row.update(manual)
    return row


def e1_table(rows_by_system, manual_by_system=None):
    manual_by_system = manual_by_system or {}
    return pd.DataFrame([e1_row(name, rows, manual_by_system.get(name))
                         for name, rows in rows_by_system.items()])


# ------------------------------------------------- blind grading of the baselines
def blind_grading_table(rows_by_system, gold, texts, seed=42):
    """Shuffle the answers of several systems into one sheet; the system is hidden.

    Returns (table for graders, key table item -> system/qid kept by the team).
    """
    from evaluation.manual_grading import build_grading_table

    parts = []
    for system, rows in rows_by_system.items():
        t = build_grading_table(rows, gold, texts)
        t.insert(0, "system", system)
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    order = list(range(len(table)))
    random.Random(seed).shuffle(order)
    table = table.iloc[order].reset_index(drop=True)
    table.insert(0, "item", [f"I{i + 1:03d}" for i in range(len(table))])
    key = table[["item", "system", "qid"]].copy()
    return table.drop(columns=["system"]), key


def manual_scores(grades_a, grades_b, key=None, system="full"):
    """Mean correctness per system (average of the two graders) from filled sheets.

    key: item -> system table from blind_grading_table (multi-system sheet). Without it the
    sheets hold one system (e.g. the day-3 sheets of the final system), named `system`.
    """
    from evaluation.manual_grading import read_grades

    a, b = read_grades(grades_a), read_grades(grades_b)
    id_col = "item" if key is not None else "qid"
    m = a[[id_col, "correctness", "citation_support"]].merge(
        b[[id_col, "correctness", "citation_support"]], on=id_col, suffixes=("_A", "_B"))
    if key is not None:
        m = m.merge(key[["item", "system"]], on="item")
    else:
        m["system"] = system
    m["correctness_avg"] = (m["correctness_A"] + m["correctness_B"]) / 2
    out = {}
    for name, g in m.groupby("system"):
        sup = pd.concat([g["citation_support_A"], g["citation_support_B"]]).dropna()
        sup = sup[sup != "n/a"]
        out[name] = {
            "manual_correctness_0_2": round(float(g["correctness_avg"].mean()), 3),
            "manual_support_yes": round(float((sup == "yes").mean()), 3) if len(sup) else None,
        }
    return out