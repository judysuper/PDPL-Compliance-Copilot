"""Evaluate BM25 and multilingual E5 + FAISS on the team's PDPL gold Q&A set.

Run from the repository root:
    python scripts/evaluate_retrieval.py

Install: pip install pandas numpy rank-bm25 faiss-cpu sentence-transformers
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import bm25_retrieval  # noqa: E402

GOLD = ROOT / "data" / "evaluation" / "gold_qa_set.csv"
INDEX_DIR = ROOT / "data" / "indexes"
OUTPUT_DIR = ROOT / "reports" / "retrieval_evaluation"
K = 5


def article_key(document_id, article_number):
    """Canonical ID includes source + article, but ignores translation language."""
    if pd.isna(document_id) or pd.isna(article_number):
        return None
    doc = str(document_id).strip().lower()
    for suffix in ("_ar", "_en"):
        if doc.endswith(suffix):
            doc = doc[:-len(suffix)]
            break
    try:
        num = str(int(float(article_number)))
    except (TypeError, ValueError, OverflowError):
        num = str(article_number).strip()
    return f"{doc}:{num}"


def assess(retrieved, expected):
    """Evaluate one question against the first K retrieved article keys."""
    top = retrieved[:K]
    expected = set(expected)
    recall = len(set(top) & expected) / len(expected)
    first_rank = next((rank for rank, key in enumerate(top, 1) if key in expected), None)
    reciprocal_rank = 1.0 / first_rank if first_rank is not None else 0.0
    return recall, reciprocal_rank, first_rank


def main():
    for p in (GOLD, INDEX_DIR / "pdpl_faiss.index", INDEX_DIR / "pdpl_metadata.csv"):
        if not p.is_file():
            raise FileNotFoundError(f"Required team file not found: {p}")
    questions = pd.read_csv(GOLD)
    required = {"question_id", "question", "language", "answerable_from_sources", "source_document", "article_number"}
    if missing := required - set(questions.columns):
        raise ValueError(f"Gold QA columns missing: {sorted(missing)}")

    answerable = questions[questions["answerable_from_sources"].astype(str).str.lower() == "true"].copy()
    if answerable.empty:
        raise ValueError("No answerable questions found")
    answerable["gold_key"] = answerable.apply(lambda r: article_key(r.source_document, r.article_number), axis=1)
    if answerable["gold_key"].isna().any():
        raise ValueError("Answerable questions missing gold source/article")

    print("Building BM25 indexes (from existing team code)...")
    bm25_indexes = bm25_retrieval.build_indexes()

    # Only load the dense model once; reuse the team's already-built FAISS index.
    import faiss
    from sentence_transformers import SentenceTransformer

    print("Loading existing FAISS index + multilingual E5 model...")
    index = faiss.read_index(str(INDEX_DIR / "pdpl_faiss.index"))
    metadata = pd.read_csv(INDEX_DIR / "pdpl_metadata.csv")
    if index.ntotal != len(metadata):
        raise ValueError(f"FAISS index ({index.ntotal}) and metadata rows ({len(metadata)}) differ")
    model = SentenceTransformer("intfloat/multilingual-e5-base")

    rows = []
    error_rows = []
    for _, item in answerable.iterrows():
        question = str(item.question)
        language = str(item.language).strip().lower()
        gold = item.gold_key

        bm25_results = bm25_retrieval.search(question, top_k=K, language=language, indexes=bm25_indexes)
        bm25_keys = [article_key(r["document_id"], r["article_number"]) for r in bm25_results]

        query_vector = model.encode(["query: " + question], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        _, ids = index.search(query_vector, K)
        dense_keys = [article_key(metadata.iloc[int(i)]["document_id"], metadata.iloc[int(i)]["article_number"]) for i in ids[0] if i >= 0]

        for method, keys in (("BM25", bm25_keys), ("Dense E5 + FAISS", dense_keys)):
            recall, mrr, first_rank = assess(keys, {gold})
            rows.append({
                "question_id": item.question_id,
                "language": language,
                "method": method,
                "gold_article": gold,
                "retrieved_top5": " | ".join(keys),
                "recall_at_5": recall,
                "reciprocal_rank_at_5": mrr,
                "first_correct_rank": first_rank,
            })
            if first_rank is None:
                error_rows.append({"question_id": item.question_id, "language": language, "method": method, "question": question, "expected": gold, "retrieved_top5": " | ".join(keys)})
        print(f"Evaluated {item.question_id} ({language})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    summary = results.groupby(["method", "language"], as_index=False).agg(
        questions=("question_id", "count"),
        recall_at_5=("recall_at_5", "mean"),
        mrr_at_5=("reciprocal_rank_at_5", "mean"),
    )
    overall = results.groupby("method", as_index=False).agg(
        questions=("question_id", "count"),
        recall_at_5=("recall_at_5", "mean"),
        mrr_at_5=("reciprocal_rank_at_5", "mean"),
    )
    overall.insert(1, "language", "overall")
    summary = pd.concat([summary, overall], ignore_index=True)
    results.to_csv(OUTPUT_DIR / "retrieval_per_question.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "retrieval_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows, columns=["question_id", "language", "method", "question", "expected", "retrieved_top5"]).to_csv(OUTPUT_DIR / "retrieval_errors.csv", index=False, encoding="utf-8-sig")
    print("\nEvaluation summary (metrics range from 0 to 1):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved three CSV reports under: {OUTPUT_DIR}")
    print("Note: scores use the single gold article currently stored per question; expand labels if other articles also support an answer.")


if __name__ == "__main__":
    main()
