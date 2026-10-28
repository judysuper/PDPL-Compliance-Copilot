"""End-to-end PDPL RAG pipeline: hybrid search -> rerank -> evidence gate -> LLM -> citation gate.

Day 3: generation now goes through generation.abstention.guarded_answer, which
uses generation.citation_checker to verify citations and refuses to answer when
the evidence or the citations are not good enough.
"""

import argparse
import json
import os
import time
from pathlib import Path

from generation.abstention import guarded_answer
from generation.reranker import Reranker
from retrieval.hybrid_retrieval import DenseSearcher, build_indexes, hybrid_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABSTENTION_CONFIG = PROJECT_ROOT / "configs" / "abstention.json"
GENERATION_CONFIG = PROJECT_ROOT / "configs" / "generation.json"

DISCLAIMER = {
    "ar": "تنبيه: هذه معلومات عامة مبنية على النصوص الرسمية المسترجعة وليست استشارة قانونية.",
    "en": "Notice: This is general information based on the retrieved official texts and is not legal advice.",
}

COMPACT_KEYS = (
    "article_id", "article_number", "source", "language", "rank", "score",
    "rrf_score", "bm25_rank", "dense_rank", "rerank_score", "rerank_rank",
)

FROM_CONFIG = "config"  # default: read the calibrated threshold from configs/abstention.json


def compact(docs):
    """Keep only ids and scores (no long article text) for logging."""
    return [{k: d[k] for k in COMPACT_KEYS if k in d} for d in docs]


def load_prompt_style(path=GENERATION_CONFIG):
    """Prompt style chosen on the calibration split, or None (then env PROMPT_STYLE / "system")."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("prompt_style")


def load_threshold(path=ABSTENTION_CONFIG):
    """Calibrated rerank-score threshold, or None if calibration has not been run."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("min_rerank_score")


class RAGPipeline:
    def __init__(self, use_reranker=True, bm25_indexes=None, dense=None, reranker=None,
                 min_rerank_score=FROM_CONFIG, prompt_style=FROM_CONFIG):
        self.bm25_indexes = bm25_indexes or build_indexes()
        self.dense = dense or DenseSearcher()
        self.reranker = reranker or (Reranker() if use_reranker else None)
        self.min_rerank_score = (
            load_threshold() if min_rerank_score == FROM_CONFIG else min_rerank_score
        )
        self.prompt_style = load_prompt_style() if prompt_style == FROM_CONFIG else prompt_style

    def retrieve(self, question, top_k=10, fused_k=20, final_k=5,
                 language=None, cross_lingual=False):
        """Retrieval + reranking only (no LLM). Reused by calibration and experiments."""
        timings = {}

        t = time.perf_counter()
        retrieval = hybrid_search(
            question, self.bm25_indexes, self.dense,
            top_k=top_k, fused_k=fused_k, language=language, cross_lingual=cross_lingual,
        )
        timings["retrieval_s"] = time.perf_counter() - t

        candidates = retrieval["fused"]
        t = time.perf_counter()
        if self.reranker is not None:
            final = self.reranker.rerank(question, candidates, top_n=final_k)
        else:
            final = candidates[:final_k]
        timings["rerank_s"] = time.perf_counter() - t

        scores = [d["rerank_score"] for d in final if "rerank_score" in d]
        return {
            "language": retrieval["language"],
            "bm25": retrieval["bm25"],
            "dense": retrieval["dense"],
            "fused": candidates,
            "final": final,
            "max_rerank_score": max(scores) if scores else None,
            "timings": timings,
        }

    def ask(
        self,
        question,
        provider=None,
        top_k=10,
        fused_k=20,
        final_k=5,
        language=None,
        cross_lingual=False,
        min_rerank_score=FROM_CONFIG,
        prompt_style=FROM_CONFIG,
        abstain_instruction=True,
        enforce_citations=True,
        max_retries=1,
        llm_fn=None,
        retrieved=None,
    ):
        """Answer a question. Pass `retrieved` (output of retrieve) to skip retrieval."""
        if retrieved is None:
            retrieved = self.retrieve(question, top_k, fused_k, final_k, language, cross_lingual)
        timings = dict(retrieved["timings"])

        threshold = self.min_rerank_score if min_rerank_score == FROM_CONFIG else min_rerank_score
        if self.reranker is None:
            threshold = None  # the threshold is calibrated on reranker scores only

        lang = retrieved["language"]
        t = time.perf_counter()
        gen = guarded_answer(
            question,
            retrieved["final"],
            provider=provider,
            language=lang,
            min_rerank_score=threshold,
            abstain_instruction=abstain_instruction,
            enforce_citations=enforce_citations,
            max_retries=max_retries,
            llm_fn=llm_fn,
            prompt_style=self.prompt_style if prompt_style == FROM_CONFIG else prompt_style,
        )
        timings["generation_s"] = time.perf_counter() - t
        timings["total_s"] = sum(v for k, v in timings.items() if k != "total_s")

        return {
            "question": question,
            "language": lang,
            "disclaimer": DISCLAIMER[lang],
            "answer": gen["answer"],
            "abstained": gen["abstained"],
            "abstention_reason": gen["abstention_reason"],
            "citation_check": gen["citation_check"],
            "first_attempt_valid": gen["first_attempt_valid"],
            "n_attempts": gen["n_attempts"],
            "raw_answer": gen["raw_answer"],
            "provider": gen["provider"],
            "llm_model": gen["llm_model"],
            "threshold": threshold,
            "prompt_style": gen["prompt_style"],
            "max_rerank_score": retrieved["max_rerank_score"],
            "bm25": compact(retrieved["bm25"]),
            "dense": compact(retrieved["dense"]),
            "fused": compact(retrieved["fused"]),
            "reranked": compact(retrieved["final"]),
            "final_sources": gen["sources_used"],
            "timings": timings,
        }


def format_result(result) -> str:
    check = result["citation_check"] or {}
    lines = [
        f"السؤال: {result['question']}",
        f"اللغة: {result['language']} | المزوّد: {result['provider']} | "
        f"النموذج: {result.get('llm_model') or '-'} | الصيغة: {result.get('prompt_style') or '-'} | "
        f"امتناع: {result['abstained']} ({result.get('abstention_reason') or '-'}) | "
        f"العتبة: {result.get('threshold')} | أعلى درجة: {result.get('max_rerank_score')}",
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
        f"فحص الاستشهادات: صالح={check.get('valid')} | المذكورة={check.get('cited_articles')} | "
        f"غير المدعومة={check.get('unsupported_articles')} | المحاولات={result.get('n_attempts')}",
        "الأزمنة (ثانية): " + ", ".join(f"{k}={v:.2f}" for k, v in result["timings"].items()),
        result["disclaimer"],
    ]
    return "\n".join(lines)


def save_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="PDPL RAG pipeline")
    parser.add_argument("question")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "ollama"))
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the calibrated rerank-score threshold")
    parser.add_argument("--prompt-style", default=None,
                        choices=["single", "system", "system_examples"],
                        help="override the prompt style chosen on the calibration split")
    args = parser.parse_args()

    pipeline = RAGPipeline(use_reranker=not args.no_rerank)
    kwargs = {} if args.threshold is None else {"min_rerank_score": args.threshold}
    if args.prompt_style:
        kwargs["prompt_style"] = args.prompt_style
    result = pipeline.ask(args.question, provider=args.provider, **kwargs)
    print(format_result(result))


if __name__ == "__main__":
    main()