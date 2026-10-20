"""End-to-end PDPL RAG pipeline: hybrid search -> rerank -> LLM -> citation check."""

import argparse
import json
import os
import time
from pathlib import Path

from generation.generation import generate_answer
from generation.reranker import Reranker
from retrieval.hybrid_retrieval import DenseSearcher, build_indexes, hybrid_search

DISCLAIMER = {
    "ar": "تنبيه: هذه معلومات عامة مبنية على النصوص الرسمية المسترجعة وليست استشارة قانونية.",
    "en": "Notice: This is general information based on the retrieved official texts and is not legal advice.",
}

COMPACT_KEYS = (
    "article_id", "article_number", "source", "language", "rank", "score",
    "rrf_score", "bm25_rank", "dense_rank", "rerank_score", "rerank_rank",
)


def compact(docs):
    """Keep only ids and scores (no long article text) for logging."""
    return [{k: d[k] for k in COMPACT_KEYS if k in d} for d in docs]


class RAGPipeline:
    def __init__(self, use_reranker=True, bm25_indexes=None, dense=None, reranker=None):
        self.bm25_indexes = bm25_indexes or build_indexes()
        self.dense = dense or DenseSearcher()
        self.reranker = reranker or (Reranker() if use_reranker else None)

    def ask(
        self,
        question,
        provider=None,
        top_k=10,
        fused_k=20,
        final_k=5,
        language=None,
        cross_lingual=False,
        min_rerank_score=None,
    ):
        timings = {}

        t = time.perf_counter()
        retrieval = hybrid_search(
            question,
            self.bm25_indexes,
            self.dense,
            top_k=top_k,
            fused_k=fused_k,
            language=language,
            cross_lingual=cross_lingual,
        )
        timings["retrieval_s"] = time.perf_counter() - t

        candidates = retrieval["fused"]

        t = time.perf_counter()
        if self.reranker is not None:
            final = self.reranker.rerank(question, candidates, top_n=final_k)
            if min_rerank_score is not None:
                final = [d for d in final if d["rerank_score"] >= min_rerank_score]
        else:
            final = candidates[:final_k]
        timings["rerank_s"] = time.perf_counter() - t

        t = time.perf_counter()
        generation = generate_answer(question, final, provider=provider)
        timings["generation_s"] = time.perf_counter() - t
        timings["total_s"] = sum(timings.values())

        lang = retrieval["language"]
        return {
            "question": question,
            "language": lang,
            "disclaimer": DISCLAIMER[lang],
            "answer": generation["answer"],
            "abstained": generation["abstained"],
            "citation_check": generation["citation_check"],
            "provider": generation["provider"],
            "bm25": compact(retrieval["bm25"]),
            "dense": compact(retrieval["dense"]),
            "fused": compact(candidates),
            "final_sources": final,
            "timings": timings,
        }


def format_result(result) -> str:
    lines = [
        f"السؤال: {result['question']}",
        f"اللغة: {result['language']} | المزوّد: {result['provider']} | امتناع: {result['abstained']}",
        "",
        result["answer"],
        "",
        "المراجع المرسلة للنموذج:",
    ]
    for i, d in enumerate(result["final_sources"], start=1):
        lines.append(
            f"  [Source {i}] المادة {d['article_number']} | {d['source']} | {d['language']}"
        )
    lines += [
        "",
        f"فحص الاستشهادات: {result['citation_check']}",
        "الأزمنة (ثانية): " + ", ".join(
            f"{k}={v:.2f}" for k, v in result["timings"].items()
        ),
        result["disclaimer"],
    ]
    return "\n".join(lines)


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="PDPL RAG pipeline")
    parser.add_argument("question")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "anthropic"))
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    pipeline = RAGPipeline(use_reranker=not args.no_rerank)
    result = pipeline.ask(args.question, provider=args.provider)
    print(format_result(result))


if __name__ == "__main__":
    main()
