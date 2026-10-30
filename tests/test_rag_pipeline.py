"""Unit tests and scenario runner for the PDPL RAG pipeline."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation.generation import (  # noqa: E402
    check_citations,
    generate_answer,
    build_context,
)
from retrieval.hybrid_retrieval import rrf_fusion  # noqa: E402


def make_doc(article_id, number, text="text", lang="ar"):
    return {
        "article_id": article_id,
        "document_id": "d1",
        "article_number": str(number),
        "article_title": "",
        "article_text": text,
        "language": lang,
        "source": "pdpl",
        "source_file": "pdpl_ar_articles.csv",
    }


# ------------------------------------------------------------------ RRF
def test_rrf_merges_duplicates_and_keeps_ranks():
    a, b, c, d = (make_doc(x, i) for i, x in enumerate("abcd", start=1))
    fused = rrf_fusion([[a, b, c], [b, d]], names=["bm25", "dense"])

    ids = [x["article_id"] for x in fused]
    assert len(ids) == len(set(ids)) == 4
    assert ids[0] == "b"  # appears high in both lists
    assert fused[0]["bm25_rank"] == 2 and fused[0]["dense_rank"] == 1
    assert fused[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)


# ------------------------------------------------------------ citations
def test_citations_valid():
    docs = [make_doc("a", 5), make_doc("b", 10)]
    check = check_citations("[Source 1] (Article 5) and [Source 2] (المادة 10)", docs)
    assert check["all_valid"]


def test_citations_invalid_source_and_article():
    docs = [make_doc("a", 5)]
    check = check_citations("[Source 3] (Article 999)", docs)
    assert check["invalid_sources"] == [3]
    assert check["unsupported_articles"] == ["999"]
    assert not check["all_valid"]


def test_citations_arabic_digits():
    docs = [make_doc("a", 5)]
    assert check_citations("[Source 1] المادة ٥", docs)["all_valid"]


def test_answer_without_citations_is_not_valid():
    assert not check_citations("إجابة بلا مصادر", [make_doc("a", 5)])["all_valid"]


# ------------------------------------------------- context / abstention
def test_context_contains_article_numbers():
    context = build_context([make_doc("a", 7, "نص المادة")])
    assert "[Source 1]" in context and "Article: 7" in context


def test_abstains_when_no_documents():
    result = generate_answer("ما هو GDPR؟", [], provider="mock")
    assert result["abstained"] is True


# ------------------------------------------------------ end-to-end smoke
INDEX_OK = (ROOT / "data" / "indexes" / "pdpl_faiss.index").exists()


@pytest.fixture(scope="module")
def pipeline():
    from rag_pipeline import RAGPipeline
    return RAGPipeline(use_reranker=False)


@pytest.mark.skipif(not INDEX_OK, reason="FAISS index not built yet")
def test_pipeline_end_to_end_mock(pipeline):
    result = pipeline.ask("ما حقوق صاحب البيانات الشخصية؟", provider="mock")
    assert result["language"] == "ar"
    assert result["final_sources"]
    assert result["citation_check"]["all_valid"]


# ------------------------------------------------------ scenario runner
SCENARIOS = [
    {"id": "ar_clear", "expect": "answer",
     "question": "ما حقوق صاحب البيانات الشخصية؟"},
    {"id": "en_clear", "expect": "answer",
     "question": "When is it permitted to process personal data without consent?"},
    {"id": "multi_article", "expect": "answer",
     "question": "What are the conditions for transferring personal data outside the Kingdom?"},
    {"id": "paraphrase_ar", "expect": "answer",
     "question": "هل يحق لي أن أطلب من الشركة حذف معلوماتي؟"},
    {"id": "article_number", "expect": "answer",
     "question": "What does Article 5 of the PDPL say?"},
    {"id": "fake_article", "expect": "abstain",
     "question": "ماذا تقول المادة 999 من نظام حماية البيانات الشخصية؟"},
    {"id": "not_covered_en", "expect": "abstain",
     "question": "What does the GDPR say about the right to be forgotten?"},
    {"id": "not_covered_ar", "expect": "abstain",
     "question": "ما نسبة ضريبة القيمة المضافة في المملكة؟"},
]


def run_scenarios(pipeline, provider=None, scenarios=SCENARIOS, out_dir=None, **ask_kwargs):
    """Run every scenario, save a CSV summary and a full JSON log."""
    out_dir = Path(out_dir or ROOT / "results" / "test_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows, full = [], []
    for s in scenarios:
        r = pipeline.ask(s["question"], provider=provider, **ask_kwargs)
        full.append({"scenario": s, "result": r})

        def top(key, n=3):
            return " | ".join(str(d["article_number"]) for d in r[key][:n])

        rows.append({
            "id": s["id"],
            "question": s["question"],
            "expect": s["expect"],
            "language": r["language"],
            "bm25_top3": top("bm25"),
            "dense_top3": top("dense"),
            "rrf_top5": top("fused", 5),
            "final_articles": " | ".join(
                str(d["article_number"]) for d in r["final_sources"]
            ),
            "abstained": r["abstained"],
            "abstain_ok": r["abstained"] == (s["expect"] == "abstain"),
            "citation_ok": (r["citation_check"] or {}).get("all_valid"),
            "cited": (r["citation_check"] or {}).get("cited_articles"),
            "total_s": round(r["timings"]["total_s"], 2),
            "answer": r["answer"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"rag_scenarios_{stamp}.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"rag_scenarios_{stamp}.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved results to {out_dir}")
    return df, full
